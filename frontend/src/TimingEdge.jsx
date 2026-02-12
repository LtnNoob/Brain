import React from 'react';
import { BaseEdge, EdgeLabelRenderer, getSmoothStepPath } from '@xyflow/react';
import { TRIGGER_COLORS } from './hooks/usePipelineConfig';

const TimingEdge = ({
  id, sourceX, sourceY, targetX, targetY,
  sourcePosition, targetPosition,
  data, selected, style: edgeStyle,
}) => {
  const triggerType = data?.triggerType || 'sequential';
  const color = TRIGGER_COLORS[triggerType] || '#6366f1';
  const delayMs = data?.delayMs || 0;
  const dataFlow = data?.dataFlow || '';
  const simActive = data?.simActive || false;

  const [edgePath, labelX, labelY] = getSmoothStepPath({
    sourceX, sourceY, targetX, targetY,
    sourcePosition, targetPosition,
    borderRadius: 12,
  });

  const isBackground = triggerType === 'background';
  const isFeedback = triggerType === 'feedback';
  const isConditional = triggerType === 'conditional';

  const strokeDasharray = isBackground ? '8 4' : isFeedback ? '4 4' : isConditional ? '12 4' : undefined;

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        style={{
          stroke: simActive ? '#fff' : color,
          strokeWidth: selected ? 2.5 : simActive ? 2 : 1.5,
          strokeDasharray,
          animation: isBackground ? 'dashmove 1s linear infinite' : undefined,
          opacity: simActive ? 1 : 0.7,
          transition: 'stroke 0.3s, opacity 0.3s',
          ...edgeStyle,
        }}
        markerEnd={`url(#marker-${triggerType})`}
      />
      <EdgeLabelRenderer>
        <div
          style={{
            position: 'absolute',
            transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
            pointerEvents: 'all',
            fontSize: 9,
            fontFamily: "'Inter', system-ui, sans-serif",
          }}
        >
          <div style={{
            background: '#0f0f1a',
            border: `1px solid ${color}40`,
            borderRadius: 4,
            padding: '2px 6px',
            color: selected ? '#fff' : '#aaa',
            whiteSpace: 'nowrap',
            display: 'flex',
            alignItems: 'center',
            gap: 4,
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%',
              background: color, display: 'inline-block', flexShrink: 0,
            }} />
            <span>{triggerType}</span>
            {delayMs > 0 && <span style={{ color: '#666' }}>+{delayMs}ms</span>}
          </div>
          {selected && dataFlow && (
            <div style={{
              marginTop: 2,
              background: '#0f0f1a',
              border: '1px solid #2a2a3e',
              borderRadius: 4,
              padding: '2px 6px',
              color: '#888',
              maxWidth: 200,
              whiteSpace: 'normal',
              lineHeight: 1.3,
            }}>
              {dataFlow}
            </div>
          )}
        </div>
      </EdgeLabelRenderer>
    </>
  );
};

// SVG marker definitions for edge arrows
export const EdgeMarkerDefs = () => (
  <svg style={{ position: 'absolute', width: 0, height: 0 }}>
    <defs>
      {Object.entries(TRIGGER_COLORS).map(([type, color]) => (
        <marker
          key={type}
          id={`marker-${type}`}
          markerWidth="8"
          markerHeight="8"
          refX="7"
          refY="4"
          orient="auto"
        >
          <path d="M 0 0 L 8 4 L 0 8 Z" fill={color} opacity="0.8" />
        </marker>
      ))}
    </defs>
  </svg>
);

export default TimingEdge;
