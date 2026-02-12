import React, { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import { CATEGORY_COLORS } from './hooks/usePipelineConfig';

const SubsystemNode = memo(({ data, selected }) => {
  const color = CATEGORY_COLORS[data.category] || '#6b7280';
  const simStatus = data.simStatus || 'idle'; // idle | running | done

  return (
    <div style={{
      ...styles.container,
      borderColor: selected ? '#fff' : (simStatus === 'running' ? color : '#2a2a3e'),
      boxShadow: selected
        ? `0 0 0 2px ${color}40`
        : simStatus === 'running'
          ? `0 0 12px ${color}60`
          : 'none',
    }}>
      <Handle type="target" position={Position.Left} style={styles.handle} />

      {/* Header */}
      <div style={{ ...styles.header, background: `${color}20`, borderBottom: `1px solid ${color}40` }}>
        <div style={styles.headerLeft}>
          <span style={{
            ...styles.statusDot,
            backgroundColor: simStatus === 'running' ? color
              : simStatus === 'done' ? '#4ade80'
              : '#444',
            animation: simStatus === 'running' ? 'pulse 1s ease-in-out infinite' : 'none',
          }} />
          <span style={{ ...styles.step, color }}>{data.step}</span>
          <span style={styles.label}>{data.label}</span>
        </div>
        <div style={{
          ...styles.toggle,
          background: data.enabled ? `${color}` : '#333',
          opacity: data.enabled ? 1 : 0.4,
        }}>
          {data.enabled ? 'ON' : 'OFF'}
        </div>
      </div>

      {/* Body */}
      <div style={styles.body}>
        <div style={styles.subsystem}>{data.subsystem}</div>
        <div style={styles.timing}>
          <span style={styles.timingIcon}>&#9202;</span>
          <span>{data.estimatedMs}ms</span>
        </div>
        {data.outputs && (
          <div style={styles.outputs}>
            {data.outputs.map((out, i) => (
              <span key={i} style={{ ...styles.outputTag, borderColor: `${color}40`, color: `${color}` }}>
                {out}
              </span>
            ))}
          </div>
        )}
      </div>

      <Handle type="source" position={Position.Right} style={styles.handle} />
    </div>
  );
});

SubsystemNode.displayName = 'SubsystemNode';

const styles = {
  container: {
    background: '#111122',
    border: '1px solid #2a2a3e',
    borderRadius: 8,
    minWidth: 180,
    maxWidth: 220,
    fontSize: 12,
    transition: 'border-color 0.2s, box-shadow 0.2s',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '6px 10px',
    borderRadius: '7px 7px 0 0',
    gap: 6,
  },
  headerLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    overflow: 'hidden',
  },
  statusDot: {
    width: 7,
    height: 7,
    borderRadius: '50%',
    flexShrink: 0,
  },
  step: {
    fontSize: 10,
    fontWeight: 700,
    flexShrink: 0,
  },
  label: {
    fontWeight: 600,
    color: '#e0e0e0',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  toggle: {
    fontSize: 9,
    fontWeight: 700,
    padding: '2px 6px',
    borderRadius: 3,
    color: '#fff',
    flexShrink: 0,
    letterSpacing: 0.5,
  },
  body: {
    padding: '8px 10px',
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
  },
  subsystem: {
    fontSize: 10,
    color: '#888',
    fontStyle: 'italic',
  },
  timing: {
    display: 'flex',
    alignItems: 'center',
    gap: 4,
    fontSize: 10,
    color: '#aaa',
  },
  timingIcon: {
    fontSize: 10,
  },
  outputs: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 3,
    marginTop: 2,
  },
  outputTag: {
    fontSize: 9,
    padding: '1px 5px',
    border: '1px solid',
    borderRadius: 3,
    whiteSpace: 'nowrap',
  },
  handle: {
    width: 8,
    height: 8,
    background: '#6366f1',
    border: '2px solid #111122',
  },
};

export default SubsystemNode;
