import { useEffect, useState } from 'react';
import { Cpu } from 'lucide-react';
import { api } from '../api';
import type { ResourceUsage } from '../types';

export function ResourceMonitor() {
  const [sample, setSample] = useState<ResourceUsage | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let disposed = false;
    let timer: number | undefined;
    let controller: AbortController;
    const poll = async () => {
      controller = new AbortController();
      const deadline = window.setTimeout(() => controller.abort(), 3000);
      try {
        const next = await api.resources(controller.signal);
        if (!disposed) { setSample(next); setFailed(false); }
      } catch {
        if (!disposed) setFailed(true);
      } finally {
        window.clearTimeout(deadline);
        if (!disposed) timer = window.setTimeout(poll, 1000);
      }
    };
    void poll();
    return () => { disposed = true; controller?.abort(); window.clearTimeout(timer); };
  }, []);
  const unavailable = failed || sample?.status === 'unavailable';
  const sampling = !failed && (!sample || sample.status === 'sampling');
  const percent = (value: number | null | undefined) => failed || value == null ? '—' : `${value.toFixed(1)}%`;
  return <div className={`cpu-monitor ${unavailable ? 'cpu-unavailable' : ''}`} role="group" aria-label="CPU 实时占用"
    title={`以整台电脑 CPU 总量为 100%；${sample?.instance_id ? '本窗口包含所属界面、后台、分析与导出子进程' : '本服务统计后台及其子进程，不包括浏览器'}。约每秒采样，当前空闲不是独享配额。`}>
    <div className="cpu-heading"><Cpu size={15} /><strong>CPU 总量 {sample?.cpu_total_percent ?? 100}%</strong>
      <span>{unavailable ? '数据不可用' : sampling ? '采样中' : `${sample?.logical_processors} 逻辑处理器`}{sample?.instance_id ? ` · 窗口 ${sample.instance_id}` : ''}</span></div>
    <dl><div><dt>整机已用</dt><dd>{percent(sample?.system_cpu_percent)}</dd></div>
      <div><dt>{sample?.instance_id ? '本窗口占用' : '本服务占用'}</dt><dd>{percent(sample?.instance_cpu_percent)}</dd></div>
      <div><dt>当前空闲</dt><dd>{percent(sample?.available_cpu_percent)}</dd></div></dl>
  </div>;
}
