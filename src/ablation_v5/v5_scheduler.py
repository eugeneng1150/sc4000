"""At most two isolated workers; measured reservations, transactional job output."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import psutil
from study_engine import GIB, digest, write_json
from study_common import folder_bytes, terminate
from v4_core import read


class Pause(Exception):
    pass


class ResourceFailure(RuntimeError):
    pass


def cpu_capacity():
    count = os.cpu_count() or 1
    try:
        count = min(count, len(os.sched_getaffinity(0)))
    except AttributeError:
        pass
    p = Path('/sys/fs/cgroup/cpu.max')
    if p.exists():
        quota, period = p.read_text().split()
        if quota != 'max':
            count = min(count, max(1, math.floor(int(quota)/int(period))))
    return max(1, count)


def available_gib():
    available = psutil.virtual_memory().available/GIB
    limit, current = Path('/sys/fs/cgroup/memory.max'), Path('/sys/fs/cgroup/memory.current')
    if limit.exists() and current.exists():
        text = limit.read_text().strip()
        if text != 'max':
            available = min(available, (int(text)-int(current.read_text()))/GIB)
    return available


def can_start(estimate, active, available, reserve, cpus, threads, maximum):
    """active entries reserve their unallocated forecast as well as current RSS."""
    if len(active) >= maximum or (len(active)+1)*threads > cpus:
        return False
    remaining = sum(max(0., a['estimate']-a['rss']) for a in active)
    return estimate+remaining <= available-reserve


def snapshot(output, reserve_gib):
    """Exclude live work; only committed results and metadata enter checkpoints."""
    output = Path(output)
    target = output.parent/(output.name+'_results.zip')
    paths = [p for p in output.rglob('*') if p.is_file() and
             '.inflight' not in p.relative_to(output).parts and not p.name.endswith('.tmp')]
    if shutil.disk_usage(output).free < sum(p.stat().st_size for p in paths)*1.05+reserve_gib*GIB:
        raise OSError('Insufficient disk for checkpoint; existing completed files remain intact')
    temporary = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for p in sorted(paths):
            z.write(p,p.relative_to(output))
    temporary.replace(target)
    return target


class Scheduler:
    def __init__(self, output, scratch, worker, cfg, deadline, heartbeat=None):
        self.output, self.scratch, self.worker = map(Path,[output,scratch,worker])
        self.cfg, self.deadline, self.heartbeat = cfg, deadline, heartbeat
        self.active = []
        self.cpus = cpu_capacity()
        self.inflight = self.output/'.inflight'
        self.inflight.mkdir(exist_ok=True)

    def event(self, **value):
        value['utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        with (self.output/'scheduler_events.jsonl').open('a') as f:
            f.write(json.dumps(value)+'\n')

    def launch(self, item):
        job,status,estimate = item['job'],Path(item['status']),item['estimate']
        status.parent.mkdir(parents=True,exist_ok=True)
        key = digest(str(status.relative_to(self.output)))[:20]
        stage = self.inflight/key
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir()
        payload = dict(job,output=str(stage))
        payload['cfg'] = dict(job['cfg'],cpu_threads=self.cfg['cpu_threads'])
        jobfile = status.with_suffix('.job.json')
        write_json(jobfile,payload)
        logpath = status.with_suffix('.log')
        log = logpath.open('w')
        env = os.environ.copy()
        for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
            env[name]=str(self.cfg['cpu_threads'])
        p = subprocess.Popen([sys.executable,str(self.worker),'--job',str(jobfile)],
                             stdout=log,stderr=subprocess.STDOUT,env=env)
        value=dict(item,process=p,log=log,stage=stage,start=time.monotonic(),
                   peak=0.,rss=0.,status=status)
        self.active.append(value)
        write_json(status,dict(status='running',pid=p.pid,estimate_gib=estimate))
        label=job.get('label',job['action'])
        print(f'RUN [{len(self.active)} worker(s)] {label} | reserved {estimate:.2f} GiB',flush=True)
        self.event(event='start',label=label,workers=len(self.active),estimate_gib=estimate,
                   available_gib=available_gib(),cpu_capacity=self.cpus,threads=self.cfg['cpu_threads'])

    def finish(self, a, reason=None):
        p = a['process']
        if p.poll() is None:
            terminate(p)
        a['log'].close()
        status = reason or ('ok' if p.returncode == 0 else 'failed')
        failure = a['stage']/'worker_failure.json'
        detail = read(failure) if failure.exists() else {}
        if status == 'failed' and detail.get('kind') == 'resource':
            status = 'memory_guard'
        if status == 'ok':
            # Commit files first and success marker last. A killed commit reruns the job.
            for file in a['stage'].iterdir():
                if file.is_file():
                    file.replace(a['status'].parent/file.name)
            shutil.rmtree(a['stage'])
        elif detail:
            write_json(a['status'].with_suffix('.failure.json'),detail)
        value=dict(status=status,peak_rss_gib=a['peak'],wall_seconds=time.monotonic()-a['start'],
                   exit_code=p.returncode,estimate_gib=a['estimate'],detail=detail)
        write_json(a['status'],value)
        self.active.remove(a)
        self.event(event='finish',label=a['job'].get('label',a['job']['action']),**value)
        print(f'{status.upper()}: {a["job"].get("label",a["job"]["action"])} '
              f'({value["wall_seconds"]/60:.1f} min, peak {a["peak"]:.2f} GiB)',flush=True)
        return status

    def batch(self, items, parallel=True):
        pending=[]
        for original in items:
            item=dict(original)
            status=Path(item['status'])
            if status.exists() and read(status).get('status') == 'ok':
                print('SKIP completed:',item['job'].get('label',status),flush=True)
                continue
            if status.exists() and read(status).get('status') in ['failed','timeout'] and not self.cfg['retry_failed']:
                raise RuntimeError(f'Previous failure: inspect {status.with_suffix(".log")}; resolve then set retry_failed')
            pending.append(item)
        if not pending:
            return
        max_workers=min(self.cfg['max_parallel'],2) if parallel else 1
        predicted=sum(i.get('fit_minutes',0.) for i in pending)
        budget=available_gib()-self.cfg['system_reserve_gib']
        forecast=list(pending);elapsed_estimate=0.;pairs=0
        while forecast:
            first=forecast.pop(0)
            duration=first.get('fit_minutes',0.)
            if max_workers==2 and 2*self.cfg['cpu_threads']<=self.cpus:
                second=next((i for i,v in enumerate(forecast)
                             if first['estimate']+v['estimate']<=budget),None)
                if second is not None:
                    duration=max(duration,forecast.pop(second).get('fit_minutes',0.))
                    pairs+=1
            elapsed_estimate+=duration
        print(f'QUEUE: {len(pending)} jobs | {pairs} estimated pair(s) fit CPU/RAM now. '
              f'Pilot-based fit work {predicted:.1f} min sequential / '
              f'~{elapsed_estimate:.1f} min with admitted pairing. '
              'Extraction, evaluation and export are additional; contention can reduce speedup.',flush=True)
        self.event(event='queue_estimate',jobs=len(pending),predicted_pairs=pairs,
                   serial_fit_minutes=predicted,paired_fit_minutes=elapsed_estimate)
        last=time.monotonic()
        try:
            while pending or self.active:
                if time.monotonic() >= self.deadline:
                    raise Pause('Session budget reached; completed jobs will be reused')
                # First collect completed work, then assess actual pressure.
                for a in list(self.active):
                    if a['process'].poll() is not None:
                        paired=len(self.active)>1
                        reason=self.finish(a)
                        if reason != 'ok':
                            if reason=='memory_guard' and paired and a.get('retries',0)<1:
                                pending.insert(0,dict(job=a['job'],status=a['status'],
                                    estimate=a['estimate'],fit_minutes=a.get('fit_minutes',0),retries=1))
                                max_workers=1
                                print('PARALLEL FALLBACK: worker memory error; retrying sequentially.',flush=True)
                                self.event(event='sequential_fallback')
                                continue
                            raise ResourceFailure(str(a['status'])) if reason=='memory_guard' else RuntimeError(
                                f'Worker failed; inspect {a["status"].with_suffix(".log")}')
                for a in list(self.active):
                    try:
                        process=psutil.Process(a['process'].pid)
                        rss=sum(p.memory_info().rss for p in [process]+process.children(recursive=True)
                                if p.is_running())/GIB
                    except psutil.NoSuchProcess:
                        continue
                    a['rss']=rss
                    a['peak']=max(a['peak'],rss)
                    action=a['job']['action']
                    minutes=self.cfg['extraction_minutes'] if action=='prepare' else (
                        self.cfg['inference_minutes'] if action=='inference' else self.cfg['job_minutes'])
                    if time.monotonic()-a['start']>minutes*60:
                        self.finish(a,'timeout')
                        raise RuntimeError(f'Job timeout; inspect {a["status"]}')
                if self.active:
                    offender=next((a for a in self.active if a['rss']>self.cfg['worker_ram_gib']),None)
                    low=available_gib()<self.cfg['system_reserve_gib']
                    if offender or low:
                        victim=offender or self.active[-1]
                        paired=len(self.active)>1
                        self.finish(victim,'memory_guard' if offender else 'system_memory_guard')
                        if paired and victim.get('retries',0)<1:
                            pending.insert(0,dict(job=victim['job'],status=victim['status'],
                                estimate=victim['estimate'],fit_minutes=victim.get('fit_minutes',0),retries=1))
                            max_workers=1
                            print('PARALLEL FALLBACK: stopped one worker; retrying it sequentially.',flush=True)
                            self.event(event='sequential_fallback')
                        else:
                            raise ResourceFailure(f'Memory threshold reached: {victim["status"]}')
                for i,item in enumerate(pending):
                    if can_start(item['estimate'],self.active,available_gib(),self.cfg['system_reserve_gib'],
                                 self.cpus,self.cfg['cpu_threads'],max_workers):
                        self.launch(pending.pop(i))
                        break
                else:
                    if pending and not self.active:
                        raise Pause('Insufficient currently available RAM for the next reservation; free memory then resume')
                if time.monotonic()-last>=30:
                    if folder_bytes(self.scratch)>self.cfg['scratch_limit_gib']*GIB:
                        raise OSError('Scratch disk cap reached')
                    if folder_bytes(self.output)>self.cfg['output_limit_gib']*GIB:
                        raise OSError('Output disk cap reached')
                    if shutil.disk_usage(self.output).free<self.cfg['disk_reserve_gib']*GIB:
                        raise OSError('Disk reserve reached')
                    print('RUNNING:', '; '.join(f'{a["job"].get("label",a["job"]["action"])} '
                        f'{(time.monotonic()-a["start"])/60:.1f} min / {a["rss"]:.2f} GiB'
                        for a in self.active),flush=True)
                    last=time.monotonic()
                if self.heartbeat:
                    self.heartbeat()
                time.sleep(self.cfg['monitor_seconds'])
        except BaseException:
            for a in list(self.active):
                self.finish(a,'interrupted')
            raise
