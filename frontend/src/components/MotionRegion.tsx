import { useId, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { ChevronRight } from 'lucide-react';

export function MotionRegion({ show, children, className = '', inline = false, id, keepMounted = false }: {
  show: boolean; children: ReactNode; className?: string; inline?: boolean; id?: string; keepMounted?: boolean;
}) {
  const [present, setPresent] = useState(show);
  const element = useRef<HTMLDivElement>(null);
  const content = useRef(children);
  if (show) content.current = children;

  useLayoutEffect(() => {
    const node = element.current;
    if (!node || (!show && !present)) return;
    if (show) setPresent(true);
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
    const horizontal = inline && getComputedStyle(node.parentElement!).flexDirection === 'row';
    const dimension = horizontal ? 'width' : 'height';
    const inner = node.firstElementChild as HTMLElement;
    const size = horizontal ? inner.getBoundingClientRect().width : inner.getBoundingClientRect().height;
    const current = node.style[dimension] || (show ? '0px' : `${size}px`);
    const opacity = node.style.opacity || (show ? '0' : '1');
    let animation: Animation | undefined;
    const finish = () => {
      animation?.cancel();
      node.style.removeProperty('height');
      node.style.removeProperty('width');
      node.style.removeProperty('opacity');
      node.removeAttribute('data-animating');
      if (!show) {
        node.style[dimension] = '0px';
        node.style.opacity = '0';
        setPresent(false);
      }
    };
    if (reduced.matches) finish();
    else {
      node.dataset.animating = 'true';
      const duration = parseFloat(getComputedStyle(node).getPropertyValue('--motion-panel')) || 220;
      animation = node.animate([
        { [dimension]: current, opacity },
        { [dimension]: show ? `${size}px` : '0px', opacity: show ? 1 : 0 },
      ], { duration, easing: 'cubic-bezier(.2,.7,.2,1)', fill: 'both' });
      animation.onfinish = finish;
    }
    const onPreference = () => { if (reduced.matches) finish(); };
    reduced.addEventListener('change', onPreference);
    return () => {
      reduced.removeEventListener('change', onPreference);
      if (animation?.playState === 'running') {
        node.style[dimension] = `${node.getBoundingClientRect()[dimension]}px`;
        node.style.opacity = getComputedStyle(node).opacity;
      }
      animation?.cancel();
    };
  }, [show, inline]);

  if (!show && !present && !keepMounted) return null;
  return <div ref={element} id={id} hidden={!show && !present} className={`motion-region ${className}`} data-open={show} inert={!show} aria-hidden={!show || undefined}>
    <div className="motion-content">{show || keepMounted ? children : content.current}</div>
  </div>;
}

export function AnimatedDisclosure({ title, children, className = '', open: controlledOpen, onToggle }: {
  title: ReactNode; children: ReactNode; className?: string; open?: boolean; onToggle?: (open: boolean) => void;
}) {
  const [localOpen, setLocalOpen] = useState(false);
  const open = controlledOpen ?? localOpen;
  const id = useId();
  return <div className={`animated-disclosure ${className}`}>
    <button type="button" className="disclosure-trigger" aria-expanded={open} aria-controls={id} onClick={() => { setLocalOpen(!open); onToggle?.(!open); }}>
      <ChevronRight size={14} aria-hidden="true" />
      <span>{title}</span>
    </button>
    <MotionRegion id={id} show={open} keepMounted>{children}</MotionRegion>
  </div>;
}
