import { useLayoutEffect, useState } from 'react';
import { flushSync } from 'react-dom';
import { Board } from './components/Board';
import type { Calibration, MoveRecord, Position } from './types';

interface RecordingPayload { initial_position: Position; records: MoveRecord[]; interval_s: number; orientation?: Calibration['orientation'] }
declare global {
  interface Window {
    loadRecording: (payload: RecordingPayload) => Promise<void>;
    renderRecordingStep: (index: number, elapsedMs: number) => number;
  }
}

export function RecordingRenderer() {
  const [payload, setPayload] = useState<RecordingPayload | null>(null);
  const [step, setStep] = useState(0);
  useLayoutEffect(() => {
    window.loadRecording = async value => {
      flushSync(() => { setPayload(value); setStep(0); });
      await document.fonts.ready;
      window.renderRecordingStep(0, 1000);
    };
    window.renderRecordingStep = (index, elapsedMs) => {
      flushSync(() => setStep(index));
      const animations = document.querySelector('.board-svg')!.getAnimations({ subtree: true });
      let duration = 0;
      for (const animation of animations) {
        duration = Math.max(duration, Number(animation.effect!.getComputedTiming().endTime));
        animation.pause();
        animation.currentTime = elapsedMs;
      }
      return duration;
    };
  }, []);
  const record = payload?.records[step - 1];
  return <div className="recording-render"><Board position={step === 0 ? payload?.initial_position ?? null : record?.position_after ?? null} orientation={payload?.orientation} step={step} lastMove={record?.move} /></div>;
}
