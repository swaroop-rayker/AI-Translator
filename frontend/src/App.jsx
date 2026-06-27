import React, { useState, useEffect } from 'react';
import { 
  LayoutDashboard, 
  Database, 
  Cpu, 
  Bookmark, 
  Terminal, 
  Sparkles, 
  CheckCircle2, 
  AlertCircle, 
  RefreshCw, 
  Play, 
  ArrowRight,
  HelpCircle,
  FileText,
  Trash2,
  Archive
} from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const INFERENCE_URL = import.meta.env.VITE_INFERENCE_URL || 'http://localhost:8080';

const formatETA = (seconds) => {
  if (seconds === null || seconds === undefined) return "Calculating...";
  if (seconds < 60) return `${seconds}s`;
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}m ${secs}s`;
};

const getAsciiProgressBar = (current, total) => {
  const pct = (!total || total <= 0) ? 0 : Math.min(100, Math.max(0, (current / total) * 100));
  const totalChars = 20;
  const filledChars = Math.round((pct / 100) * totalChars);
  const emptyChars = totalChars - filledChars;
  
  return (
    <span>
      <span style={{ color: '#34d399' }}>{"█".repeat(Math.max(0, filledChars))}</span>
      <span style={{ color: 'rgba(255, 255, 255, 0.12)' }}>{"░".repeat(Math.max(0, emptyChars))}</span>
      <span style={{ color: '#34d399', fontWeight: 'bold' }}>{` ${Math.round(pct)}%`}</span>
    </span>
  );
};

// Beautiful SVG Chart Component
function SVGLineChart({ data, title, color = '#8b5cf6', yLabel = 'Loss' }) {
  if (!data || data.length === 0) {
    return (
      <div style={{ height: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8' }}>
        No metric data available yet. Start training to see live curves!
      </div>
    );
  }

  const width = 500;
  const height = 180;
  const padding = 30;

  const minX = 0;
  const maxX = data.length - 1;
  const minY = Math.min(...data) * 0.9;
  const maxY = Math.max(...data) * 1.1;

  const getX = (index) => padding + (index / (data.length - 1 || 1)) * (width - 2 * padding);
  const getY = (val) => height - padding - ((val - minY) / (maxY - minY || 1)) * (height - 2 * padding);

  let pathD = `M ${getX(0)} ${getY(data[0])}`;
  for (let i = 1; i < data.length; i++) {
    pathD += ` L ${getX(i)} ${getY(data[i])}`;
  }

  let fillD = `${pathD} L ${getX(data.length - 1)} ${height - padding} L ${getX(0)} ${height - padding} Z`;

  return (
    <div style={{ width: '100%' }}>
      <h5 style={{ fontSize: '14px', marginBottom: '8px', color: '#94a3b8', fontWeight: 500 }}>{title}</h5>
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto', background: 'rgba(0,0,0,0.2)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
        {/* Gradients */}
        <defs>
          <linearGradient id={`grad-${title.replace(/\s+/g, '')}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.3"/>
            <stop offset="100%" stopColor={color} stopOpacity="0.0"/>
          </linearGradient>
        </defs>
        
        {/* Grid Lines */}
        <line x1={padding} y1={padding} x2={width-padding} y2={padding} stroke="rgba(255,255,255,0.05)" />
        <line x1={padding} y1={height/2} x2={width-padding} y2={height/2} stroke="rgba(255,255,255,0.05)" />
        <line x1={padding} y1={height-padding} x2={width-padding} y2={height-padding} stroke="rgba(255,255,255,0.1)" />
        
        {/* Fill Area */}
        <path d={fillD} fill={`url(#grad-${title.replace(/\s+/g, '')})`} />
        
        {/* Line Path */}
        <path d={pathD} fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
        
        {/* Data Points (circle dots) */}
        {data.map((val, idx) => (
          <circle 
            key={idx} 
            cx={getX(idx)} 
            cy={getY(val)} 
            r={data.length > 20 ? "1.5" : "3.5"} 
            fill={color} 
            stroke="#101628" 
            strokeWidth="1.5" 
          />
        ))}

        {/* Y Axis labels */}
        <text x={padding - 5} y={padding + 5} fill="#64748b" fontSize="9" textAnchor="end">{maxY.toFixed(2)}</text>
        <text x={padding - 5} y={height - padding + 3} fill="#64748b" fontSize="9" textAnchor="end">{minY.toFixed(2)}</text>
        <text x={padding + 5} y={height - 5} fill="#64748b" fontSize="9">Start</text>
        <text x={width - padding - 5} y={height - 5} fill="#64748b" fontSize="9" textAnchor="end">Step {data.length}</text>
      </svg>
    </div>
  );
}

// TrainingProgressPanel Component for stage-by-stage and telemetry monitoring
function TrainingProgressPanel({ progress, jobStatus, errorLog, isCompact = false, jobId, onCancel, onPause, onResume }) {
  const stages = [
    { key: 'init', label: 'Init' },
    { key: 'loading_dataset', label: 'Data Load' },
    { key: 'preprocessing_dataset', label: 'Tokenize' },
    { key: 'loading_model', label: 'VRAM Load' },
    { key: 'training', label: 'Training' },
    { key: 'finalizing', label: 'Finalize' }
  ];

  const currentStage = progress?.stage || (jobStatus === 'Queued' ? 'queue' : 'init');
  const stageProgress = progress?.stage_progress !== undefined ? progress.stage_progress : 0;
  const stageDetails = progress?.stage_details || (jobStatus === 'Queued' ? 'Enqueued in Celery task worker. Waiting for GPU lock release...' : 'Initializing trainer...');

  const activeIdx = stages.findIndex(s => s.key === currentStage);
  
  let totalPercent = 0;
  if (currentStage === 'queue') {
    totalPercent = 0;
  } else if (activeIdx >= 0) {
    const basePercent = (activeIdx / stages.length) * 100;
    const currentShare = (stageProgress / stages.length);
    totalPercent = Math.min(100, Math.round(basePercent + currentShare));
  } else {
    totalPercent = stageProgress ? Math.round(stageProgress) : 0;
  }

  const speed = progress?.samples_per_sec;
  const eta = progress?.eta;
  const epoch = progress?.epoch;
  const totalEpochs = progress?.total_epochs;
  const step = progress?.step;
  const totalSteps = progress?.total_steps;
  const loss = progress?.loss;
  const valLoss = progress?.val_loss;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', width: '100%' }}>
      {/* Stepper Pipeline */}
      {!isCompact && (
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center', 
          background: 'rgba(0,0,0,0.2)', 
          padding: '10px 14px', 
          borderRadius: '8px', 
          border: '1px solid rgba(255,255,255,0.03)',
          overflowX: 'auto',
          gap: '8px',
          marginBottom: '4px'
        }}>
          {stages.map((st, idx) => {
            const isCompleted = idx < activeIdx;
            const isActive = st.key === currentStage;
            
            let color = '#64748b'; 
            let dotBg = 'rgba(255,255,255,0.02)';
            if (isCompleted) {
              color = 'var(--color-success)';
              dotBg = 'rgba(16, 185, 129, 0.1)';
            } else if (isActive) {
              color = '#60a5fa';
              dotBg = 'rgba(96, 165, 250, 0.15)';
            }

            return (
              <div key={st.key} style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0 }}>
                <div style={{ 
                  width: '20px', 
                  height: '20px', 
                  borderRadius: '50%', 
                  background: dotBg, 
                  border: `1px solid ${isActive ? '#60a5fa' : isCompleted ? 'var(--color-success)' : 'rgba(255,255,255,0.08)'}`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '10px',
                  fontWeight: 'bold',
                  color: color,
                  boxShadow: isActive ? '0 0 6px rgba(96, 165, 250, 0.2)' : 'none'
                }}>
                  {isCompleted ? '✓' : idx + 1}
                </div>
                <span style={{ fontSize: '11px', fontWeight: isActive ? 600 : 400, color: isActive ? '#f1f5f9' : color }}>
                  {st.label}
                </span>
                {idx < stages.length - 1 && (
                  <span style={{ color: 'rgba(255,255,255,0.06)', marginLeft: '4px' }}>➔</span>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Progress Card */}
      <div style={{ 
        background: 'rgba(255,255,255,0.005)', 
        border: '1px solid rgba(255,255,255,0.03)', 
        padding: '12px', 
        borderRadius: '8px',
        display: 'flex',
        flexDirection: 'column',
        gap: '10px'
      }}>
        {/* Stage & Details */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <span className={`badge ${currentStage === 'queue' ? 'warning' : 'info'}`} style={{ textTransform: 'uppercase', fontSize: '8.5px', fontWeight: 700, padding: '1px 5px', letterSpacing: '0.05em' }}>
              {currentStage.replace('_', ' ')}
            </span>
            <div style={{ fontSize: '11.5px', color: '#e2e8f0', marginTop: '4px', fontWeight: 500, lineHeight: '1.4' }}>
              {stageDetails}
            </div>
            {progress?.current_value !== undefined && progress?.total_value !== undefined && progress.current_value !== null && progress.total_value !== null && (
              <div style={{ fontSize: '10.5px', color: '#94a3b8', marginTop: '3px' }}>
                Processed: <strong>{progress.current_value.toLocaleString()}</strong> / <strong>{progress.total_value.toLocaleString()}</strong> pairs
              </div>
            )}
          </div>
          <span style={{ fontSize: '13px', fontWeight: 700, color: '#60a5fa' }}>
            {totalPercent}%
          </span>
        </div>

        {/* ASCII representation */}
        <pre style={{ 
          margin: 0, 
          padding: '8px 10px', 
          background: 'rgba(0, 0, 0, 0.3)', 
          borderRadius: '5px', 
          fontFamily: 'monospace', 
          fontSize: isCompact ? '10px' : '12px', 
          color: '#34d399', 
          border: '1px solid rgba(255, 255, 255, 0.04)',
          overflowX: 'auto',
          whiteSpace: 'pre'
        }}>
          <div>Progress</div>
          {getAsciiProgressBar(totalPercent, 100)}
        </pre>

        {/* Graphic progress bar */}
        <div style={{ height: '4px', background: 'rgba(255,255,255,0.04)', borderRadius: '2px', overflow: 'hidden' }}>
          <div style={{ 
            height: '100%', 
            width: `${totalPercent}%`, 
            background: 'linear-gradient(90deg, #8b5cf6 0%, #60a5fa 100%)', 
            transition: 'width 0.4s ease-out' 
          }}></div>
        </div>

        {/* Telemetry/Metrics Grid */}
        {currentStage === 'training' && (
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: isCompact ? '1fr 1fr' : '1fr 1fr 1fr', 
            gap: '8px', 
            marginTop: '2px',
            borderTop: '1px solid rgba(255,255,255,0.04)',
            paddingTop: '8px'
          }}>
            <div style={{ background: 'rgba(0,0,0,0.1)', padding: '6px 10px', borderRadius: '5px', border: '1px solid rgba(255,255,255,0.01)' }}>
              <div style={{ fontSize: '9px', color: '#64748b', textTransform: 'uppercase', marginBottom: '1px' }}>Epoch</div>
              <div style={{ fontSize: '13px', fontWeight: 600 }}>{epoch} / {totalEpochs}</div>
            </div>
            <div style={{ background: 'rgba(0,0,0,0.1)', padding: '6px 10px', borderRadius: '5px', border: '1px solid rgba(255,255,255,0.01)' }}>
              <div style={{ fontSize: '9px', color: '#64748b', textTransform: 'uppercase', marginBottom: '1px' }}>Step</div>
              <div style={{ fontSize: '13px', fontWeight: 600 }}>{step} / {totalSteps}</div>
            </div>
            <div style={{ background: 'rgba(0,0,0,0.1)', padding: '6px 10px', borderRadius: '5px', border: '1px solid rgba(255,255,255,0.01)' }}>
              <div style={{ fontSize: '9px', color: '#64748b', textTransform: 'uppercase', marginBottom: '1px' }}>Loss (Current / Val)</div>
              <div style={{ fontSize: '12px', fontWeight: 600 }}>
                <span style={{ color: '#f87171' }}>{loss?.toFixed(4) || 'N/A'}</span>
                {" / "}
                <span style={{ color: '#60a5fa' }}>{valLoss != null ? valLoss.toFixed(4) : 'Pending'}</span>
              </div>
            </div>
            <div style={{ background: 'rgba(0,0,0,0.1)', padding: '6px 10px', borderRadius: '5px', border: '1px solid rgba(255,255,255,0.01)' }}>
              <div style={{ fontSize: '9px', color: '#64748b', textTransform: 'uppercase', marginBottom: '1px' }}>Speed</div>
              <div style={{ fontSize: '12.5px', fontWeight: 600, color: '#22d3ee' }}>
                {speed ? `${speed} samples/s` : 'Calculating...'}
              </div>
            </div>
            <div style={{ background: 'rgba(0,0,0,0.1)', padding: '6px 10px', borderRadius: '5px', border: '1px solid rgba(255,255,255,0.01)' }}>
              <div style={{ fontSize: '9px', color: '#64748b', textTransform: 'uppercase', marginBottom: '1px' }}>ETA</div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: '#34d399' }}>{formatETA(eta)}</div>
            </div>
          </div>
        )}
      </div>

      {/* Action buttons inside the progress panel */}
      {jobId && (
        <div style={{ display: 'flex', gap: '8px', marginTop: '4px' }}>
          {/* Stop / Remove button */}
          {onCancel && (jobStatus === 'Running' || jobStatus === 'Starting' || jobStatus === 'Queued' || jobStatus === 'Paused') && (
            <button 
              className="secondary" 
              style={{ 
                padding: '6px 10px', 
                fontSize: '11.5px', 
                background: 'rgba(239, 68, 68, 0.1)', 
                color: '#fca5a5', 
                border: '1px solid rgba(239, 68, 68, 0.2)',
                flex: 1,
                cursor: 'pointer',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '6px',
                borderRadius: '6px'
              }}
              onClick={(e) => {
                e.stopPropagation();
                const confirmMsg = jobStatus === 'Queued' 
                  ? "Are you sure you want to remove this job from the training queue?" 
                  : "Are you sure you want to stop this training run? Paused checkpoints will be deleted.";
                if (window.confirm(confirmMsg)) {
                  onCancel(jobId);
                }
              }}
            >
              {jobStatus === 'Queued' ? '🛑 Remove from Queue' : '🛑 Stop Training'}
            </button>
          )}

          {/* Pause button */}
          {onPause && (jobStatus === 'Running' || jobStatus === 'Starting' || jobStatus === 'Queued') && (
            <button 
              className="secondary" 
              style={{ 
                padding: '6px 10px', 
                fontSize: '11.5px', 
                background: 'rgba(245, 158, 11, 0.1)', 
                color: '#fcd34d', 
                border: '1px solid rgba(245, 158, 11, 0.2)',
                flex: 1,
                cursor: 'pointer',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '6px',
                borderRadius: '6px'
              }}
              onClick={(e) => {
                e.stopPropagation();
                onPause(jobId);
              }}
            >
              ⏸ Pause
            </button>
          )}

          {/* Resume button */}
          {onResume && jobStatus === 'Paused' && (
            <button 
              style={{ 
                padding: '6px 10px', 
                fontSize: '11.5px', 
                background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)', 
                color: '#ffffff', 
                border: 'none',
                flex: 1,
                cursor: 'pointer',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '6px',
                borderRadius: '6px',
                boxShadow: 'none'
              }}
              onClick={(e) => {
                e.stopPropagation();
                onResume(jobId);
              }}
            >
              ▶ Resume
            </button>
          )}
        </div>
      )}

      {/* Robust Error & Troubleshooting Guide */}
      {jobStatus === 'Failed' && (
        <div style={{ 
          background: 'rgba(239, 68, 68, 0.04)', 
          border: '1px solid rgba(239, 68, 68, 0.15)', 
          padding: '12px', 
          borderRadius: '8px', 
          fontSize: '11.5px', 
          color: '#fca5a5',
          display: 'flex',
          flexDirection: 'column',
          gap: '8px'
        }}>
          <div style={{ fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span>✕ Training Execution Failed</span>
          </div>
          {errorLog ? (
            <pre style={{ 
              margin: 0, padding: '8px', background: 'rgba(0,0,0,0.3)', 
              borderRadius: '5px', fontSize: '10.5px', color: '#fca5a5', 
              overflowX: 'auto', maxHeight: '140px', fontFamily: 'monospace', 
              whiteSpace: 'pre-wrap', border: '1px solid rgba(255,255,255,0.03)'
            }}>
              {errorLog}
            </pre>
          ) : (
            <div>An unexpected error occurred during model training. Click view logs for details.</div>
          )}
          <div style={{ 
            fontSize: '10.5px', 
            color: '#94a3b8', 
            background: 'rgba(255,255,255,0.01)', 
            padding: '8px', 
            borderRadius: '5px',
            borderLeft: '2.5px solid #8b5cf6'
          }}>
            <strong>Troubleshooting Checklist:</strong>
            <ul style={{ paddingLeft: '14px', marginTop: '3px', display: 'flex', flexDirection: 'column', gap: '2px' }}>
              <li><strong>VRAM Out of Memory:</strong> Enable FP16 (Mixed Precision), reduce batch_size to 1 or 2, or shorten max_sequence_length to 64.</li>
              <li><strong>Celery Worker Offline:</strong> Verify host celery training worker is running via terminal with `--pool=solo`.</li>
              <li><strong>Redis/DB Status:</strong> Check docker service container logs.</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

// ParentCard Component
function ParentCard({ ver, activeMerge, fetchPreview, handleDelete, handleSubsetSubmit, index }) {
  const isMerged = ver.processing_history?.is_merged;
  const isProcessing = ver.status === 'Processing';
  const [isAutoOffset, setIsAutoOffset] = useState(true);
  
  // Find active merge progress
  const isCurrentMerging = activeMerge && activeMerge.datasetId === ver.dataset_id && activeMerge.phase === 'merging';
  const progressPercent = isCurrentMerging ? Math.min(100, ((activeMerge.processedCount / (activeMerge.totalToProcess || 1)) * 100)) : 0;
  
  let badgeColor = isMerged ? 'info' : 'muted';
  let badgeText = isMerged ? 'MERGED' : 'RAW';
  
  if (isProcessing) {
    badgeColor = 'warning';
    badgeText = 'MERGING';
  }

  return (
    <div 
      className="glass-panel" 
      style={{ 
        padding: '16px', 
        borderLeft: `4px solid ${isProcessing ? '#f59e0b' : 'var(--color-primary)'}`, 
        background: 'rgba(255,255,255,0.01)',
        borderRadius: '12px',
        display: 'flex',
        flexDirection: 'column',
        gap: '10px'
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h5 style={{ fontSize: '13.5px', fontWeight: 700, color: '#e2e8f0', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '70%' }} title={ver.version}>
          <span style={{ color: 'var(--color-primary)', marginRight: '6px' }}>#{index}</span>
          {ver.version}
        </h5>
        <span className={`badge ${badgeColor}`} style={{ fontSize: '9px', padding: '1px 6px', fontWeight: isProcessing ? 700 : undefined }}>
          {isProcessing ? '● ' : ''}{badgeText}
        </span>
      </div>

      <p style={{ fontSize: '12px', color: '#94a3b8', lineHeight: '1.4' }}>
        Lang: <strong>{ver.src_lang} ➔ {ver.tgt_lang}</strong><br />
        Records: <strong>{isProcessing ? (
          isCurrentMerging && activeMerge.linesMerged > 0
            ? `Merging (${activeMerge.linesMerged.toLocaleString()} lines)`
            : 'Merging...'
        ) : ver.record_count.toLocaleString()}</strong>
      </p>

      {isProcessing && (
        <div style={{ background: 'rgba(255,255,255,0.02)', padding: '10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.04)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#94a3b8', marginBottom: '6px' }}>
            <span>Merging Files</span>
            <span>{isCurrentMerging ? `${progressPercent.toFixed(0)}%` : '0%'}</span>
          </div>
          <div style={{ height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', overflow: 'hidden' }}>
            <div 
              style={{ 
                height: '100%', 
                width: `${isCurrentMerging ? progressPercent : 0}%`, 
                background: 'linear-gradient(90deg, #f59e0b 0%, #d97706 100%)',
                transition: 'width 0.4s ease-out'
              }}
            ></div>
          </div>
        </div>
      )}

      {isMerged && ver.processing_history?.merged_from && (
        <div style={{ background: 'rgba(255,255,255,0.02)', padding: '6px 8px', borderRadius: '4px', border: '1px solid rgba(255,255,255,0.03)' }}>
          <p style={{ fontSize: '10px', color: '#a78bfa', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={`Source: ${ver.processing_history.merged_from.src}\nTarget: ${ver.processing_history.merged_from.tgt}`}>
            From: <code>{(ver.processing_history.merged_from.src || "").split(/[/\\\\]/).pop() || ""}</code> + <code>{(ver.processing_history.merged_from.tgt || "").split(/[/\\\\]/).pop() || ""}</code>
          </p>
        </div>
      )}

      <div style={{ fontSize: '10px', color: '#64748b', fontFamily: 'monospace' }}>
        <div>Disk: <span style={{ color: '#94a3b8' }}>{(ver.storage_path || '').replace(/\\/g, '/')}</span></div>
        <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>Hash: {(ver.file_hash || '').slice(0, 16)}...</div>
      </div>

      <div style={{ display: 'flex', gap: '6px', marginTop: '4px' }}>
        <button className="secondary" style={{ padding: '4px 8px', fontSize: '11px', flex: 1 }} onClick={() => fetchPreview(ver.id)} disabled={isProcessing}>
          Preview
        </button>
        <button 
          className="secondary" 
          style={{ padding: '4px 8px', fontSize: '11px', background: 'rgba(239, 68, 68, 0.05)', color: '#fca5a5', border: '1px solid rgba(239, 68, 68, 0.1)', flex: 1 }} 
          onClick={() => handleDelete(ver.id)}
        >
          {isProcessing ? 'Force Delete' : 'Delete'}
        </button>
      </div>

      {ver.record_count > 0 && !isProcessing && (
        <div style={{ marginTop: '6px', paddingTop: '8px', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
          {ver.record_count >= 100000 && (
            <div style={{ fontSize: '10.5px', color: '#fcd34d', background: 'rgba(245, 158, 11, 0.05)', border: '1px solid rgba(245, 158, 11, 0.15)', padding: '4px 6px', borderRadius: '4px', marginBottom: '6px', lineHeight: '1.3' }}>
              ⚠️ Large corpus ({ver.record_count.toLocaleString()} rows). Extract a batched subset before quality cleaning.
            </div>
          )}
          <div style={{ fontSize: '11px', color: '#38bdf8', marginBottom: '8px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span>Next auto-offset: <strong>{(ver.processing_history?.next_subset_offset || 0).toLocaleString()}</strong></span>
            <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', margin: 0, fontWeight: 500 }}>
              <input 
                id={`subset-auto-offset-${ver.id}`}
                type="checkbox"
                checked={isAutoOffset}
                onChange={(e) => setIsAutoOffset(e.target.checked)}
                style={{ cursor: 'pointer', margin: 0, width: '13px', height: '13px' }}
              />
              Auto Offset
            </label>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', background: 'rgba(255, 255, 255, 0.02)', padding: '10px', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.04)' }}>
            <div style={{ display: 'flex', gap: '8px' }}>
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <span style={{ fontSize: '10px', color: '#94a3b8', fontWeight: 500 }}>Size (Lines)</span>
                <input 
                  id={`subset-size-${ver.id}`} 
                  type="number" 
                  defaultValue={50000} 
                  style={{ padding: '6px 8px', fontSize: '11px', height: '28px', background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '4px', color: '#f8fafc' }}
                  title="Subset Size"
                />
              </div>
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <span style={{ fontSize: '10px', color: '#94a3b8', fontWeight: 500 }}>Strategy</span>
                <select id={`subset-strategy-${ver.id}`} style={{ padding: '4px 8px', fontSize: '11px', height: '28px', background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '4px', color: '#f8fafc', width: '100%' }}>
                  <option value="first_n">Sequential</option>
                  {!isAutoOffset && <option value="random_sample">Random</option>}
                </select>
              </div>
            </div>

            {!isAutoOffset && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <span style={{ fontSize: '10px', color: '#94a3b8', fontWeight: 500 }}>Start Offset</span>
                <input 
                  id={`subset-offset-${ver.id}`} 
                  type="number" 
                  defaultValue={0} 
                  style={{ padding: '6px 8px', fontSize: '11px', height: '28px', background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '4px', color: '#f8fafc' }}
                  title="Start Line Offset (Sequential Strategy only)"
                />
              </div>
            )}

            <button 
              className="btn" 
              style={{ width: '100%', padding: '6px 12px', fontSize: '11px', height: '28px', background: 'linear-gradient(135deg, #06b6d4 0%, #0891b2 100%)', borderRadius: '4px', fontWeight: 600, marginTop: '2px', boxShadow: 'none' }} 
              onClick={() => handleSubsetSubmit(ver.id)}
            >
              Extract Batched Subset
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// BatchedCard Component
function BatchedCard({ ver, activeMerge, fetchPreview, handleDelete, triggerProcessing, cancelPipeline, cancelJob, isSelected, onSelectToggle, jobs, index }) {
  const queuedJob = jobs && jobs.find(j => j && j.job_type === 'dataset_processing' && j.status === 'Queued' && j.config?.dataset_version_id === ver.id);
  const runningJob = jobs && jobs.find(j => j && j.job_type === 'dataset_processing' && j.status === 'Running' && j.config?.dataset_version_id === ver.id);
  const failedJob = jobs && jobs.find(j => j && j.job_type === 'dataset_processing' && j.status === 'Failed' && j.config?.dataset_version_id === ver.id);

  const isCurrentCleaning = (activeMerge && activeMerge.isCleaning && 
    (activeMerge.versionId === ver.id || (activeMerge.datasetId === ver.dataset_id && activeMerge.phase !== 'ended' && activeMerge.status === 'processing'))) || !!runningJob;

  let badgeColor = 'muted';
  let badgeText = 'READY';

  if (queuedJob) {
    badgeColor = 'warning';
    badgeText = 'QUEUED';
  } else if (runningJob || isCurrentCleaning) {
    badgeColor = 'info';
    badgeText = 'CLEANING';
  } else if (ver.status === 'Processed' || ver.status === 'TrainReady' || ver.status === 'TrainingUsed') {
    badgeColor = 'success';
    badgeText = 'ENDED';
  } else if (ver.status === 'Validated') {
    badgeColor = 'muted';
    badgeText = 'READY';
  } else if (ver.status === 'Cancelled') {
    badgeColor = 'warning';
    badgeText = 'CANCELLED';
  } else if (ver.status === 'Failed' || failedJob) {
    badgeColor = 'danger';
    badgeText = 'FAILED';
  }

  const hasEnded = badgeText === 'ENDED';

  return (
    <div 
      className="glass-panel" 
      style={{ 
        padding: '16px', 
        borderLeft: '4px solid #22d3ee', 
        background: 'rgba(255,255,255,0.01)',
        borderRadius: '12px',
        display: 'flex',
        flexDirection: 'column',
        gap: '10px'
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', maxWidth: '70%' }}>
          <input 
            type="checkbox" 
            checked={isSelected || false} 
            onChange={onSelectToggle} 
            style={{ width: '14px', height: '14px', cursor: 'pointer', margin: 0 }}
            disabled={queuedJob || runningJob}
          />
          <h5 style={{ fontSize: '13.5px', fontWeight: 700, color: '#e2e8f0', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', margin: 0 }} title={ver.version}>
            <span style={{ color: '#22d3ee', marginRight: '6px' }}>#{index}</span>
            {ver.version}
          </h5>
        </div>
        <span className={`badge ${badgeColor}`} style={{ fontSize: '9px', padding: '1px 6px', fontWeight: 700 }}>
          ● {badgeText}
        </span>
      </div>

      <p style={{ fontSize: '12px', color: '#94a3b8', lineHeight: '1.4' }}>
        Lang: <strong>{ver.src_lang} ➔ {ver.tgt_lang}</strong><br />
        Records: <strong>{ver.record_count.toLocaleString()}</strong>
      </p>

      {isCurrentCleaning && activeMerge && activeMerge.status === 'processing' && (
        <div style={{ marginTop: '4px', background: 'rgba(255,255,255,0.02)', padding: '10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.04)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#94a3b8', marginBottom: '6px' }}>
            <span>Cleaning Progress</span>
            <span>{Math.min(100, ((activeMerge.processedCount / (activeMerge.totalToProcess || 1)) * 100)).toFixed(0)}%</span>
          </div>
          <div style={{ height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', overflow: 'hidden', marginBottom: '8px' }}>
            <div 
              style={{ 
                height: '100%', 
                width: `${Math.min(100, (activeMerge.processedCount / (activeMerge.totalToProcess || 1)) * 100)}%`, 
                background: 'linear-gradient(90deg, #22d3ee 0%, #06b6d4 100%)',
                transition: 'width 0.4s ease-out'
              }}
            ></div>
          </div>
          <span style={{ fontSize: '10.5px', color: '#64748b' }}>
            Processed: {activeMerge.processedCount.toLocaleString()} / {activeMerge.totalToProcess.toLocaleString()} records
          </span>
        </div>
      )}

      <div style={{ fontSize: '10px', color: '#64748b', fontFamily: 'monospace' }}>
        <div>Disk: <span style={{ color: '#94a3b8' }}>{(ver.storage_path || '').replace(/\\/g, '/')}</span></div>
        <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>Hash: {(ver.file_hash || '').slice(0, 16)}...</div>
      </div>

      <div style={{ display: 'flex', gap: '6px', marginTop: '4px' }}>
        <button className="secondary" style={{ padding: '4px 8px', fontSize: '11px', flex: 1 }} onClick={() => fetchPreview(ver.id)}>
          Preview
        </button>
        
        {queuedJob ? (
          <button 
            className="secondary" 
            style={{ padding: '4px 8px', fontSize: '11px', background: 'rgba(239, 68, 68, 0.08)', color: '#fca5a5', border: '1px solid rgba(239, 68, 68, 0.15)', flex: 1.5 }}
            onClick={() => {
              if (confirm("Are you sure you want to cancel this queued cleaning job?")) {
                cancelJob(queuedJob.id);
              }
            }}
          >
            Cancel Job
          </button>
        ) : runningJob || isCurrentCleaning ? (
          <button 
            className="secondary" 
            style={{ padding: '4px 8px', fontSize: '11px', background: 'rgba(239, 68, 68, 0.1)', color: '#fca5a5', border: '1px solid rgba(239, 68, 68, 0.2)', flex: 1.5 }}
            onClick={() => {
              if (runningJob) {
                if (confirm("Are you sure you want to stop this dataset cleaning job?")) {
                  cancelJob(runningJob.id);
                }
              } else {
                cancelPipeline(ver.dataset_id);
              }
            }}
          >
            Stop Pipeline
          </button>
        ) : !hasEnded ? (
          <button 
            style={{ padding: '4px 8px', fontSize: '11px', background: 'linear-gradient(135deg, #06b6d4 0%, #0891b2 100%)', boxShadow: 'none', flex: 1.5 }}
            onClick={() => triggerProcessing(ver.id, ver.dataset_id, ver.version)}
          >
            Clean & Ingest
          </button>
        ) : (
          <button 
            className="secondary" 
            style={{ padding: '4px 8px', fontSize: '11px', background: 'rgba(239, 68, 68, 0.05)', color: '#fca5a5', border: '1px solid rgba(239, 68, 68, 0.1)', flex: 1 }} 
            onClick={() => handleDelete(ver.id)}
          >
            Delete
          </button>
        )}
      </div>
    </div>
  );
}

// CleanedCard Component
function CleanedCard({ ver, experiments, jobs, models, inferenceStatus, fetchPreview, handleLoadInTrainer, handleArchive, handleDelete, deployModel, isSelected, onSelectToggle, index, cancelJob, resetFailedJob, pauseJob, resumeJob }) {
  const [showErrorLog, setShowErrorLog] = useState(false);

  const safeExperiments = Array.isArray(experiments) ? experiments : [];
  const safeJobs = Array.isArray(jobs) ? jobs : [];
  const safeModels = Array.isArray(models) ? models : [];

  // 1. Check if there is an active running experiment for this version
  const activeRun = safeExperiments.find(e => e && e.dataset_version_id === ver.id && (e.status === 'Running' || e.status === 'Starting'));
  // 2. Check if there is a queued/running/paused training job in Celery
  const queuedJob = safeJobs.find(j => j && j.job_type === 'training' && (j.status === 'Queued' || j.status === 'Running') && j.config?.dataset_version_id === ver.id);
  const pausedJob = safeJobs.find(j => j && j.job_type === 'training' && j.status === 'Paused' && j.config?.dataset_version_id === ver.id);
  
  // Determine state
  let trainingState = 'READY'; // READY, IN_QUEUE, IN_TRAINING, PAUSED, TRAINED
  let activeJobOrRun = null;

  if (activeRun) {
    trainingState = 'IN_TRAINING';
    activeJobOrRun = activeRun;
  } else if (queuedJob) {
    trainingState = queuedJob.status === 'Queued' ? 'IN_QUEUE' : 'IN_TRAINING';
    activeJobOrRun = queuedJob;
  } else if (pausedJob) {
    trainingState = 'PAUSED';
    activeJobOrRun = pausedJob;
  } else if (safeExperiments.some(e => e && e.dataset_version_id === ver.id && e.status === 'Completed') || ver.status === 'TrainingUsed') {
    trainingState = 'TRAINED';
  }

  // Check the most recent job and run status for this version to prevent stale/historical runs from locking the card state
  const versionJobs = safeJobs.filter(j => j && j.job_type === 'training' && j.config?.dataset_version_id === ver.id);
  const latestJob = versionJobs[0] || null;
  
  const versionRuns = safeExperiments.filter(e => e && e.dataset_version_id === ver.id);
  const latestRun = versionRuns[0] || null;

  const isFailed = (latestJob && latestJob.status === 'Failed') || (latestRun && latestRun.status === 'Failed');
  const failedJob = isFailed ? latestJob : null;
  const failedRun = isFailed ? latestRun : null;

  if (isFailed) {
    trainingState = 'FAILED';
  }

  // Check if this version is registered as a model
  const registeredModel = safeModels.find(m => m && m.dataset_version_id === ver.id);
  const isServed = registeredModel && inferenceStatus && inferenceStatus.model_version === registeredModel.version;

  // State badge helper
  let badgeColor = 'success';
  let badgeText = 'READY';
  if (trainingState === 'IN_QUEUE') {
    badgeColor = 'warning';
    badgeText = 'IN QUEUE';
  } else if (trainingState === 'IN_TRAINING') {
    badgeColor = 'info';
    badgeText = 'IN TRAINING';
  } else if (trainingState === 'PAUSED') {
    badgeColor = 'warning';
    badgeText = 'PAUSED';
  } else if (trainingState === 'TRAINED') {
    badgeColor = 'success';
    badgeText = 'TRAINED';
  } else if (trainingState === 'FAILED') {
    badgeColor = 'danger';
    badgeText = 'FAILED';
  }

  return (
    <div 
      className="glass-panel" 
      style={{ 
        padding: '16px', 
        borderLeft: `4px solid ${trainingState === 'TRAINED' ? '#10b981' : trainingState === 'IN_TRAINING' ? '#f59e0b' : trainingState === 'PAUSED' ? '#fbbf24' : trainingState === 'FAILED' ? '#ef4444' : '#34d399'}`, 
        background: ver.status === 'Archived' ? 'rgba(255,255,255,0.005)' : 'rgba(255,255,255,0.01)',
        borderRadius: '12px',
        opacity: ver.status === 'Archived' ? 0.6 : 1,
        display: 'flex',
        flexDirection: 'column',
        gap: '10px'
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', maxWidth: '70%' }}>
          <input 
            type="checkbox" 
            checked={isSelected || false} 
            onChange={onSelectToggle} 
            style={{ width: '14px', height: '14px', cursor: 'pointer', margin: 0 }}
            disabled={trainingState === 'IN_TRAINING'}
          />
          <h5 style={{ fontSize: '13.5px', fontWeight: 700, color: '#e2e8f0', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', margin: 0 }} title={ver.version}>
            <span style={{ color: '#34d399', marginRight: '6px' }}>#{index}</span>
            {ver.version}
          </h5>
        </div>
        <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
          {ver.status === 'Archived' && (
            <span className="badge muted" style={{ fontSize: '8px', padding: '1px 4px' }}>ARCHIVED</span>
          )}
          <span className={`badge ${badgeColor}`} style={{ fontSize: '9px', padding: '1px 6px', fontWeight: 700 }}>
            ● {badgeText}
          </span>
        </div>
      </div>

      <p style={{ fontSize: '12px', color: '#94a3b8', lineHeight: '1.4' }}>
        Lang: <strong>{ver.src_lang} ➔ {ver.tgt_lang}</strong><br />
        Records: <strong>{ver.record_count.toLocaleString()}</strong>
      </p>

      {/* Display Model Info if Trained */}
      {trainingState === 'TRAINED' && registeredModel && (
        <div style={{ background: 'rgba(16, 185, 129, 0.05)', border: '1px solid rgba(16, 185, 129, 0.15)', padding: '10px', borderRadius: '8px', fontSize: '12px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
            <strong style={{ color: '#a7f3d0' }}>Checkpoint: {registeredModel.version}</strong>
            {isServed && (
              <span className="badge success" style={{ fontSize: '8px', padding: '1px 4px', background: 'rgba(16, 185, 129, 0.2)', border: '1px solid #10b981', color: '#34d399', display: 'flex', alignItems: 'center', gap: '2px' }}>
                🚀 SERVED
              </span>
            )}
          </div>
          <div style={{ color: '#94a3b8', fontSize: '11px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px' }}>
            <div>Loss: <strong>{registeredModel.metrics?.final_loss?.toFixed(4) || 'N/A'}</strong></div>
            <div>VRAM: <strong>{registeredModel.metrics?.model_size_mb || 836} MB</strong></div>
          </div>
          {registeredModel.approval_status === 'Approved' && !isServed && (
            <button 
              className="secondary" 
              style={{ width: '100%', marginTop: '8px', padding: '4px 8px', fontSize: '11px', background: 'rgba(16, 185, 129, 0.1)', color: '#34d399', border: '1px solid rgba(16, 185, 129, 0.2)' }}
              onClick={() => deployModel(registeredModel.id)}
            >
              Deploy / Serve Model
            </button>
          )}
        </div>
      )}

      {/* Training states info */}
      {(trainingState === 'IN_QUEUE' || trainingState === 'IN_TRAINING' || trainingState === 'PAUSED') && (() => {
        const trainingJob = safeJobs.find(j => j && j.job_type === 'training' && (j.status === 'Running' || j.status === 'Starting' || j.status === 'Queued' || j.status === 'Paused') && j.config?.dataset_version_id === ver.id);
        return (
          <TrainingProgressPanel 
            progress={trainingJob?.config?.progress} 
            jobStatus={trainingJob?.status || 'Running'} 
            errorLog={trainingJob?.error_log} 
            jobId={trainingJob?.id}
            onCancel={cancelJob}
            onPause={pauseJob}
            onResume={resumeJob}
            isCompact={true} 
          />
        );
      })()}

      {/* Failure state */}
      {isFailed && (
        <div style={{ background: 'rgba(239, 68, 68, 0.05)', border: '1px solid rgba(239, 68, 68, 0.15)', padding: '8px', borderRadius: '6px', fontSize: '11.5px', color: '#fca5a5', display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>✕ Training Run Failed</span>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button 
                style={{ background: 'transparent', border: 'none', color: '#3b82f6', textDecoration: 'underline', padding: 0, fontSize: '11px', cursor: 'pointer', boxShadow: 'none' }}
                onClick={() => setShowErrorLog(!showErrorLog)}
              >
                {showErrorLog ? 'Hide Log' : 'View Log'}
              </button>
              {failedJob && (
                <button 
                  style={{ background: 'transparent', border: 'none', color: '#34d399', textDecoration: 'underline', padding: 0, fontSize: '11px', cursor: 'pointer', boxShadow: 'none', fontWeight: 600 }}
                  onClick={() => resetFailedJob(failedJob.id)}
                >
                  Dismiss & Reset
                </button>
              )}
            </div>
          </div>
          {showErrorLog && (
            <pre style={{ margin: 0, padding: '6px', background: 'rgba(0,0,0,0.3)', borderRadius: '4px', fontSize: '10px', color: '#fca5a5', overflowX: 'auto', maxHeight: '120px', fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
              {failedJob?.error_log || failedRun?.metrics?.error || (failedJob?.status === 'Failed' ? "Training job failed or was aborted by user. Check host celery terminal logs." : "No error log details available.")}
            </pre>
          )}
        </div>
      )}

      <div style={{ fontSize: '10px', color: '#64748b', fontFamily: 'monospace' }}>
        <div>Disk: <span style={{ color: '#94a3b8' }}>{(ver.storage_path || '').replace(/\\/g, '/')}</span></div>
        <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>Hash: {(ver.file_hash || '').slice(0, 16)}...</div>
      </div>

      <div style={{ display: 'flex', gap: '6px', marginTop: '4px' }}>
        <button className="secondary" style={{ padding: '4px 8px', fontSize: '11px', flex: 1 }} onClick={() => fetchPreview(ver.id)}>
          Preview
        </button>
        
        {trainingState === 'READY' && (
          <button 
            style={{ padding: '4px 8px', fontSize: '11px', background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)', boxShadow: 'none', flex: 1.5 }}
            onClick={() => handleLoadInTrainer(ver.id)}
          >
            Train Model ➔
          </button>
        )}

        {trainingState === 'TRAINED' && (
          <button 
            className="secondary" 
            style={{ padding: '4px 8px', fontSize: '11px', border: '1px solid rgba(245, 158, 11, 0.2)', color: '#fcd34d', flex: 1.5 }}
            onClick={() => handleLoadInTrainer(ver.id)}
          >
            Retrain / Tune
          </button>
        )}

        {ver.status !== 'Archived' && (trainingState === 'READY' || trainingState === 'TRAINED') && (
          <button 
            className="secondary" 
            style={{ padding: '4px 8px', width: '28px', flex: '0 0 auto', display: 'flex', alignItems: 'center', justifyContent: 'center' }} 
            title="Archive Version"
            onClick={() => handleArchive(ver.id)}
          >
            <Archive size={12} />
          </button>
        )}

        {trainingState === 'READY' && (
          <button 
            className="secondary" 
            style={{ padding: '4px 8px', width: '28px', flex: '0 0 auto', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(239, 68, 68, 0.05)', color: '#fca5a5', border: '1px solid rgba(239, 68, 68, 0.1)' }} 
            title="Delete Version"
            onClick={() => handleDelete(ver.id)}
          >
            <Trash2 size={12} />
          </button>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [selectedBatches, setSelectedBatches] = useState([]);
  const [selectedCleaned, setSelectedCleaned] = useState([]);
  const [showFlowVisualizer, setShowFlowVisualizer] = useState(false);
  const [showEngineInfo, setShowEngineInfo] = useState(true);
  const [activeConsoleTab, setActiveConsoleTab] = useState('ledger');
  const [gpuStatus, setGpuStatus] = useState({ is_locked: false, active_job_id: null, redis_connected: false, gpu_util: 0, vram_util: 0, cpu_util: 0 });
  const [datasets, setDatasets] = useState([]);
  const safeDatasets = Array.isArray(datasets) ? datasets : [];
  const [jobs, setJobs] = useState([]);
  const activeTrainingJob = Array.isArray(jobs) ? jobs.find(j => j && j.job_type === 'training' && (j.status === 'Running' || j.status === 'Starting' || j.status === 'Queued' || j.status === 'Paused')) : null;
  const [experiments, setExperiments] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedRunIds, setSelectedRunIds] = useState([]);
  const [comparison, setComparison] = useState(null);
  const [showArchived, setShowArchived] = useState(false);

  // Form inputs
  const [datasetName, setDatasetName] = useState('');
  const [srcLang, setSrcLang] = useState('en');
  const [tgtLang, setTgtLang] = useState('kn');
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadMode, setUploadMode] = useState('single'); // 'single' or 'moses'
  const [srcFilePath, setSrcFilePath] = useState('');
  const [tgtFilePath, setTgtFilePath] = useState('');
  
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [trainConfig, setTrainConfig] = useState({
    model_name: 'facebook/mbart-large-50-many-to-many-mmt',
    epochs: 3,
    batch_size: 4,
    learning_rate: 0.00005,
    max_sequence_length: 128,
    fp16: true
  });

  // Moses active merge tracker
  const [activeMerge, setActiveMerge] = useState(null);

  // Preview data
  const [previewData, setPreviewData] = useState(null);
  const [selectedVersionForPreview, setSelectedVersionForPreview] = useState(null);

  // Translation Sandbox
  const [sandboxRequest, setSandboxRequest] = useState({ text: '', src: 'en', tgt: 'kn' });
  const [sandboxResult, setSandboxResult] = useState(null);
  const [sandboxLoading, setSandboxLoading] = useState(false);
  const [inferenceStatus, setInferenceStatus] = useState(null);

  // Fetch initial data
  const refreshAll = () => {
    fetch(`${API_URL}/api/jobs/status`).then(r => r.json()).then(setGpuStatus).catch(err => console.log("Offline"));
    fetch(`${API_URL}/api/jobs`).then(r => r.json()).then(setJobs).catch(err => {});
    fetch(`${API_URL}/api/experiments`).then(r => r.json()).then(setExperiments).catch(err => {});
    fetch(`${API_URL}/api/models`).then(r => r.json()).then(setModels).catch(err => {});
    fetch(`${INFERENCE_URL}/status`).then(r => r.json()).then(setInferenceStatus).catch(err => {});
    fetch(`http://localhost:8001/datasets`).then(r => r.json()).then(setDatasets).catch(err => {});
  };

  useEffect(() => {
    refreshAll();
    const interval = setInterval(refreshAll, 5000);
    return () => clearInterval(interval);
  }, []);

  // Sync dataset versions list from jobs or registry
  useEffect(() => {
    // Collect dataset versions from experiment runs & models to display in dropdowns
    const list = [];
    if (Array.isArray(experiments)) {
      experiments.forEach(e => {
        if (e && e.dataset_version && e.dataset_version_id && !list.some(d => d && d.id === e.dataset_version_id)) {
          list.push({ id: e.dataset_version_id, version: e.dataset_version });
        }
      });
    }
    // Add default mock versions if empty to guide user
    if (list.length === 0) {
      list.push({ id: 'mock-dataset-v4', version: 'Kannada corpus train ready (v4_train_ready)' });
    }
  }, [experiments]);

  // Handle file upload / merge ingestion
  const handleUpload = (e) => {
    e.preventDefault();
    if (uploadMode === 'single') {
      if (!selectedFile) {
        alert("Please select a file to upload.");
        return;
      }
      
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('name', datasetName);
      formData.append('src_lang', srcLang);
      formData.append('tgt_lang', tgtLang);

      fetch(`http://localhost:8001/datasets/upload`, {
        method: 'POST',
        body: formData
      })
      .then(r => {
        if (!r.ok) return r.json().then(err => { throw new Error(err.detail || "Upload failed"); });
        return r.json();
      })
      .then(data => {
        setActiveMerge({
          datasetId: data.dataset_id,
          name: data.name,
          processedCount: 0,
          totalToProcess: 0,
          status: 'processing',
          phase: 'ingesting',
          isMerged: false
        });
        refreshAll();
      })
      .catch(err => alert("Upload failed: " + err.message));
    } else {
      if (!srcFilePath || !tgtFilePath) {
        alert("Please enter both source and target file paths.");
        return;
      }

      const formData = new FormData();
      formData.append('name', datasetName);
      formData.append('src_lang', srcLang);
      formData.append('tgt_lang', tgtLang);
      formData.append('src_path_input', srcFilePath);
      formData.append('tgt_path_input', tgtFilePath);

      fetch(`http://localhost:8001/datasets/merge-moses`, {
        method: 'POST',
        body: formData
      })
      .then(r => {
        if (!r.ok) return r.json().then(err => { throw new Error(err.detail || "Merge failed"); });
        return r.json();
      })
      .then(data => {
        setActiveMerge({
          datasetId: data.dataset_id,
          name: data.name,
          processedCount: 0,
          totalToProcess: 0,
          status: 'processing',
          phase: 'merging',
          isMerged: true
        });
        refreshAll();
      })
      .catch(err => alert("Merge failed: " + err.message));
    }
  };

  // Auto-detect stuck/running merges on page load or datasets update
  useEffect(() => {
    const processingDataset = safeDatasets.find(d => d && d.status === 'Processing');
    if (processingDataset && !activeMerge) {
      setActiveMerge({
        datasetId: processingDataset.dataset_id,
        versionId: processingDataset.id,
        name: processingDataset.version,
        status: 'processing',
        processedCount: 0,
        totalToProcess: 0,
        phase: 'merging',
        isMerged: true
      });
    }
  }, [safeDatasets, activeMerge]);

  // Poll for Moses merge progress
  useEffect(() => {
    if (!activeMerge || activeMerge.status === 'completed' || activeMerge.status === 'failed' || activeMerge.status === 'cancelled' || activeMerge.status === 'ended') return;

    const interval = setInterval(() => {
      fetch(`http://localhost:8001/datasets/merge-status/${activeMerge.datasetId}`)
      .then(r => r.json())
      .then(data => {
        setActiveMerge(prev => {
          if (!prev) return null;
          return {
            ...prev,
            processedCount: parseInt(data.processed_count) || 0,
            totalToProcess: parseInt(data.total_to_process) || 0,
            linesMerged: parseInt(data.lines_merged) || 0,
            status: data.status,
            phase: data.phase,
            error: data.error
          };
        });
        
        if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled' || data.status === 'ended') {
          refreshAll();
          // clear progress bar card after 8 seconds
          setTimeout(() => setActiveMerge(null), 8000);
        }
      })
      .catch(err => {});
    }, 2000);

    return () => clearInterval(interval);
  }, [activeMerge]);

  // Cancel running pipeline
  const cancelPipeline = (datasetId) => {
    fetch(`http://localhost:8001/datasets/merge-cancel/${datasetId}`, { method: 'POST' })
    .then(r => {
      if (!r.ok) throw new Error("Failed to cancel pipeline");
      return r.json();
    })
    .then(() => {
      setActiveMerge(prev => prev ? { ...prev, status: 'cancelled', error: 'Cancellation requested. Cleaning up data...' } : null);
      refreshAll();
    })
    .catch(err => alert("Failed to cancel pipeline: " + err.message));
  };

  // Cancel platform job (dataset processing or model training)
  const cancelJob = (jobId) => {
    fetch(`${API_URL}/api/jobs/${jobId}/cancel`, { method: 'POST' })
    .then(r => {
      if (!r.ok) throw new Error("Failed to cancel job");
      return r.json();
    })
    .then(() => {
      refreshAll();
    })
    .catch(err => alert("Failed to cancel job: " + err.message));
  };

  // Pause training job
  const pauseJob = (jobId) => {
    fetch(`${API_URL}/api/jobs/${jobId}/pause`, { method: 'POST' })
    .then(r => {
      if (!r.ok) throw new Error("Failed to pause job");
      return r.json();
    })
    .then(() => {
      refreshAll();
    })
    .catch(err => alert("Failed to pause job: " + err.message));
  };

  // Resume training job
  const resumeJob = (jobId) => {
    fetch(`${API_URL}/api/jobs/${jobId}/resume`, { method: 'POST' })
    .then(r => {
      if (!r.ok) throw new Error("Failed to resume job");
      return r.json();
    })
    .then(() => {
      refreshAll();
    })
    .catch(err => alert("Failed to resume job: " + err.message));
  };

  // Reset failed platform job and restore dataset version to TrainReady
  const resetFailedJob = (jobId) => {
    fetch(`${API_URL}/api/jobs/${jobId}/reset`, { method: 'POST' })
    .then(r => {
      if (!r.ok) throw new Error("Failed to reset job");
      return r.json();
    })
    .then(() => {
      refreshAll();
    })
    .catch(err => alert("Failed to reset job: " + err.message));
  };

  // Purge all failed and cancelled jobs and experiment runs
  const handlePurgeFailed = () => {
    if (!window.confirm("Are you sure you want to delete all failed and cancelled training runs/jobs? Stale dataset cards will be reset, and successful runs will be kept.")) {
      return;
    }
    fetch(`${API_URL}/api/jobs/purge-failed`, { method: 'POST' })
    .then(r => {
      if (!r.ok) throw new Error("Failed to purge failed runs");
      return r.json();
    })
    .then(data => {
      alert(`Successfully purged ${data.purged_jobs_count} failed jobs, ${data.purged_runs_count} failed runs, and restored ${data.reverted_datasets_count} datasets.`);
      refreshAll();
    })
    .catch(err => alert("Failed to purge runs: " + err.message));
  };

  // Trigger Validation
  const triggerValidation = (verId) => {
    fetch(`http://localhost:8001/datasets/${verId}/validate`, { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      alert(`Validation complete! Status: ${data.status}`);
      refreshAll();
    });
  };

  // Trigger Cleaning Processing (routed to Orchestrator Celery queue)
  const triggerProcessing = (verId, datasetId, datasetName) => {
    fetch(`${API_URL}/api/jobs/dataset-process?dataset_version_id=${verId}&min_length=2&max_length=150`, { method: 'POST' })
    .then(r => {
      if (!r.ok) return r.json().then(err => { throw new Error(err.detail || "Processing failed"); });
      return r.json();
    })
    .then(data => {
      alert("Job submitted to Celery queue! It will process sequentially.");
      refreshAll();
    })
    .catch(err => alert("Failed to start processing: " + err.message));
  };

  // Trigger Dataset Subsetting (Automated Batching with offset support)
  const triggerSubsetting = (verId, size, strategy, offset) => {
    fetch(`http://localhost:8001/datasets/${verId}/subset?max_records=${size}&strategy=${strategy}&line_offset=${offset}`, { method: 'POST' })
    .then(r => {
      if (!r.ok) return r.json().then(err => { throw new Error(err.detail || "Subsetting failed"); });
      return r.json();
    })
    .then(data => {
      alert(data.message || "Subsetting complete!");
      refreshAll();
    })
    .catch(err => alert("Failed to subset dataset: " + err.message));
  };

  const handleSubsetSubmit = (verId) => {
    const sizeInput = document.getElementById(`subset-size-${verId}`);
    const strategyInput = document.getElementById(`subset-strategy-${verId}`);
    const offsetInput = document.getElementById(`subset-offset-${verId}`);
    const autoOffsetInput = document.getElementById(`subset-auto-offset-${verId}`);
    const size = sizeInput ? parseInt(sizeInput.value) : 50000;
    const strategy = strategyInput ? strategyInput.value : 'first_n';
    const isAuto = autoOffsetInput ? autoOffsetInput.checked : false;
    const offset = isAuto ? -1 : (offsetInput ? parseInt(offsetInput.value) : 0);
    triggerSubsetting(verId, size, strategy, offset);
  };

  // Fetch Preview (toggles close if already active on the same version)
  const fetchPreview = async (verId) => {
    if (selectedVersionForPreview === verId) {
      setPreviewData(null);
      setSelectedVersionForPreview(null);
      return;
    }
    setSelectedVersionForPreview(verId);
    try {
      const response = await fetch(`http://localhost:8001/datasets/${verId}/preview?limit=10`);
      if (!response.ok) {
        let errorMsg = `HTTP error ${response.status}`;
        try {
          const errData = await response.json();
          if (errData && errData.detail) {
            errorMsg = errData.detail;
          }
        } catch (e) {
          // Keep fallback error message if JSON parsing fails
        }
        throw new Error(errorMsg);
      }
      const data = await response.json();
      if (data && Array.isArray(data.records)) {
        setPreviewData(data.records);
      } else {
        setPreviewData([]);
      }
    } catch (err) {
      alert(`Failed to fetch preview: ${err.message}`);
      setPreviewData(null);
      setSelectedVersionForPreview(null);
    }
  };

  // Submit training job
  const handleTrainSubmit = (e) => {
    e.preventDefault();
    if (!selectedDatasetId) {
      alert("Please select a train-ready dataset version!");
      return;
    }
    
    const params = new URLSearchParams({
      dataset_version_id: selectedDatasetId,
      model_name: trainConfig.model_name,
      epochs: trainConfig.epochs.toString(),
      batch_size: trainConfig.batch_size.toString(),
      learning_rate: trainConfig.learning_rate.toString(),
      max_sequence_length: trainConfig.max_sequence_length.toString()
    });

    fetch(`${API_URL}/api/jobs/train?${params.toString()}`, { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      alert(data.message || "Training job queued!");
      refreshAll();
    })
    .catch(err => alert("Failed to submit training job."));
  };

  // Model Registry Actions
  const approveModel = (id) => {
    fetch(`${API_URL}/api/models/${id}/approve`, { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      alert("Model Approved!");
      refreshAll();
    });
  };

  const deployModel = (id) => {
    fetch(`${API_URL}/api/models/${id}/deploy`, { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      alert("Model Deployed to Inference Service!");
      refreshAll();
    });
  };

  // Run Comparisons
  const handleCompare = () => {
    if (selectedRunIds.length === 0) return;
    const query = selectedRunIds.map(id => `ids=${id}`).join('&');
    fetch(`${API_URL}/api/experiments/compare?${query}`)
    .then(r => r.json())
    .then(setComparison)
    .catch(err => alert("Error comparing runs"));
  };

  // Run Inference Sandbox translation
  const handleTranslate = (e) => {
    e.preventDefault();
    if (!sandboxRequest.text.strip()) return;
    setSandboxLoading(true);
    
    fetch(`${INFERENCE_URL}/translate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: sandboxRequest.text,
        src_lang: sandboxRequest.src,
        tgt_lang: sandboxRequest.tgt
      })
    })
    .then(r => r.json())
    .then(data => {
      setSandboxResult(data);
      setSandboxLoading(false);
    })
    .catch(err => {
      setSandboxResult({ error: "Translation request failed. Make sure inference service is online." });
      setSandboxLoading(false);
    });
  };

  // Purge/Reset MLOps platform data
  const handlePurge = () => {
    if (!window.confirm("ARE YOU ABSOLUTELY SURE? This will permanently delete all datasets, versions, experiments, models, and files on disk!")) return;
    
    fetch(`http://localhost:8001/datasets/purge`, { method: 'POST' })
    .then(r => {
      if (!r.ok) return r.json().then(err => { throw new Error(err.detail || "Purge failed"); });
      return r.json();
    })
    .then(data => {
      alert(data.message || "Platform reset successfully!");
      setActiveMerge(null);
      refreshAll();
    })
    .catch(err => alert("Reset failed: " + err.message));
  };

  // Pre-load a clean version into trainer and switch tabs
  const handleLoadInTrainer = (verId) => {
    setSelectedDatasetId(verId);
    setActiveTab('training');
  };

  const handleArchive = (verId) => {
    if (!window.confirm("Are you sure you want to archive this dataset version? It will be hidden from the active columns.")) return;
    fetch(`http://localhost:8001/datasets/${verId}/archive`, { method: 'POST' })
    .then(r => {
      if (!r.ok) throw new Error("Archive failed");
      return r.json();
    })
    .then(() => {
      refreshAll();
    })
    .catch(err => alert("Failed to archive dataset: " + err.message));
  };

  const handleDelete = (verId) => {
    if (!window.confirm("Are you sure you want to PERMANENTLY delete this version and its files on disk? This cannot be undone.")) return;
    fetch(`http://localhost:8001/datasets/${verId}`, { method: 'DELETE' })
    .then(r => {
      if (!r.ok) throw new Error("Delete failed");
      return r.json();
    })
    .then(() => {
      // Clear deleted ID from selection states if present
      setSelectedBatches(prev => prev.filter(id => id !== verId));
      setSelectedCleaned(prev => prev.filter(id => id !== verId));
      refreshAll();
    })
    .catch(err => alert("Failed to delete dataset: " + err.message));
  };

  const handleBulkDelete = (verIds, description) => {
    if (verIds.length === 0) return;
    if (!window.confirm(`Are you sure you want to PERMANENTLY delete ${verIds.length} version(s) of ${description} and their files on disk? This cannot be undone.`)) return;
    
    fetch(`http://localhost:8001/datasets/bulk-delete`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ version_ids: verIds })
    })
    .then(r => {
      if (!r.ok) throw new Error("Bulk delete failed");
      return r.json();
    })
    .then(data => {
      alert(data.message || "Bulk deletion completed.");
      setSelectedBatches(prev => prev.filter(id => !verIds.includes(id)));
      setSelectedCleaned(prev => prev.filter(id => !verIds.includes(id)));
      refreshAll();
    })
    .catch(err => alert("Bulk delete failed: " + err.message));
  };

  const handleBulkClean = (versionIds) => {
    if (!versionIds || versionIds.length === 0) return;
    if (!window.confirm(`Are you sure you want to clean and ingest ${versionIds.length} selected batches sequentially?`)) return;
    
    // Submit each to Orchestrator dataset-process endpoint
    const promises = versionIds.map(verId => 
      fetch(`${API_URL}/api/jobs/dataset-process?dataset_version_id=${verId}&min_length=2&max_length=150`, { method: 'POST' })
        .then(r => {
          if (!r.ok) return r.json().then(err => { throw new Error(verId + ": " + (err.detail || "Failed")); });
          return r.json();
        })
    );
    
    Promise.all(promises)
      .then(() => {
        alert(`Successfully scheduled ${versionIds.length} cleaning jobs in Celery. They will execute sequentially.`);
        setSelectedBatches([]); // Clear selection
        refreshAll();
      })
      .catch(err => {
        alert("Some jobs failed to schedule: " + err.message);
        refreshAll();
      });
  };

  const renderEmptyColumn = (message) => (
    <div style={{ padding: '24px', textAlign: 'center', color: '#64748b', fontSize: '12px', border: '1px dashed rgba(255,255,255,0.05)', borderRadius: '8px', width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      {message}
    </div>
  );



  return (
    <div className="dashboard-layout">
      {/* Sidebar Navigation */}
      <div className="sidebar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '32px', padding: '0 8px' }}>
          <Sparkles color="#8b5cf6" size={24} />
          <h2 style={{ fontSize: '18px', fontWeight: 800, tracking: '-0.5px' }} className="gradient-text">TRANSLATE.AI</h2>
        </div>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', flex: 1 }}>
          <div 
            onClick={() => setActiveTab('dashboard')} 
            style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px', borderRadius: '8px', cursor: 'pointer', background: activeTab === 'dashboard' ? 'rgba(139, 92, 246, 0.15)' : 'transparent', color: activeTab === 'dashboard' ? '#8b5cf6' : '#94a3b8', fontWeight: 550, transition: 'all 0.2s' }}
          >
            <LayoutDashboard size={18} /> Dashboard
          </div>
          
          <div 
            onClick={() => setActiveTab('datasets')} 
            style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px', borderRadius: '8px', cursor: 'pointer', background: activeTab === 'datasets' ? 'rgba(139, 92, 246, 0.15)' : 'transparent', color: activeTab === 'datasets' ? '#8b5cf6' : '#94a3b8', fontWeight: 550, transition: 'all 0.2s' }}
          >
            <Database size={18} /> Datasets
          </div>
          
          <div 
            onClick={() => setActiveTab('training')} 
            style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px', borderRadius: '8px', cursor: 'pointer', background: activeTab === 'training' ? 'rgba(139, 92, 246, 0.15)' : 'transparent', color: activeTab === 'training' ? '#8b5cf6' : '#94a3b8', fontWeight: 550, transition: 'all 0.2s' }}
          >
            <Cpu size={18} /> Training & Runs
          </div>
          
          <div 
            onClick={() => setActiveTab('registry')} 
            style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px', borderRadius: '8px', cursor: 'pointer', background: activeTab === 'registry' ? 'rgba(139, 92, 246, 0.15)' : 'transparent', color: activeTab === 'registry' ? '#8b5cf6' : '#94a3b8', fontWeight: 550, transition: 'all 0.2s' }}
          >
            <Bookmark size={18} /> Model Registry
          </div>
          
          <div 
            onClick={() => setActiveTab('sandbox')} 
            style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px', borderRadius: '8px', cursor: 'pointer', background: activeTab === 'sandbox' ? 'rgba(139, 92, 246, 0.15)' : 'transparent', color: activeTab === 'sandbox' ? '#8b5cf6' : '#94a3b8', fontWeight: 550, transition: 'all 0.2s' }}
          >
            <Terminal size={18} /> Sandbox
          </div>
        </div>

        {/* GPU Scheduler Card */}
        <div className="glass-panel" style={{ padding: '16px', background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.03)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '12px', color: '#94a3b8' }}>GPU Hardware</span>
            <span className={`badge ${gpuStatus.is_locked ? 'warning' : 'success'}`} style={{ padding: '2px 6px', fontSize: '9px' }}>
              {gpuStatus.is_locked ? 'Busy' : 'Idle'}
            </span>
          </div>
          <p style={{ fontSize: '11px', color: '#64748b' }}>
            {gpuStatus.is_locked ? `Job Active: ${(gpuStatus.active_job_id || '').slice(0,8)}` : 'RTX 4050 Scheduler Available'}
          </p>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="main-content">
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '40px' }}>
          <div>
            <span style={{ fontSize: '12px', color: '#8b5cf6', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '1px' }}>AI translation development suite</span>
            <h1 style={{ fontSize: '28px', fontWeight: 800, marginTop: '4px' }}>
              {activeTab === 'dashboard' && "Overview Dashboard"}
              {activeTab === 'datasets' && "Dataset Processing & Versioning"}
              {activeTab === 'training' && "Training Jobs & MLflow Tracker"}
              {activeTab === 'registry' && "Model Checkpoint Registry"}
              {activeTab === 'sandbox' && "Inference Testing Sandbox"}
            </h1>
          </div>
          <div style={{ display: 'flex', gap: '12px' }}>
            {activeTab === 'datasets' && (
              <button 
                onClick={handlePurge}
                style={{ 
                  background: 'rgba(239, 68, 68, 0.1)', 
                  border: '1px solid rgba(239, 68, 68, 0.25)', 
                  color: '#fca5a5',
                  boxShadow: 'none'
                }}
              >
                Reset & Purge All
              </button>
            )}
            <button className="secondary" onClick={refreshAll}>
              <RefreshCw size={16} /> Sync Live
            </button>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'dashboard' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
            {/* Stats grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '20px' }}>
              <div className="glass-panel">
                <span style={{ fontSize: '13px', color: '#94a3b8' }}>Active GPU Scheduler</span>
                <h3 style={{ fontSize: '24px', fontWeight: 700, marginTop: '8px' }}>{gpuStatus.is_locked ? '1 Active' : '0 Running'}</h3>
                <span style={{ fontSize: '11px', color: '#64748b' }}>Limit 1 job (RTX 4050)</span>
              </div>
              <div className="glass-panel">
                <span style={{ fontSize: '13px', color: '#94a3b8' }}>Experiment Runs</span>
                <h3 style={{ fontSize: '24px', fontWeight: 700, marginTop: '8px' }}>{experiments.length} Total</h3>
                <span style={{ fontSize: '11px', color: '#64748b' }}>MLflow Metadata tracking</span>
              </div>
              <div className="glass-panel">
                <span style={{ fontSize: '13px', color: '#94a3b8' }}>Registered Models</span>
                <h3 style={{ fontSize: '24px', fontWeight: 700, marginTop: '8px' }}>{models.length} Models</h3>
                <span style={{ fontSize: '11px', color: '#64748b' }}>Only approved can deploy</span>
              </div>
              <div className="glass-panel">
                <span style={{ fontSize: '13px', color: '#94a3b8' }}>Stateless Inference Status</span>
                <h3 style={{ fontSize: '24px', fontWeight: 700, marginTop: '8px', color: '#10b981' }}>
                  {inferenceStatus ? 'Active' : 'Offline'}
                </h3>
                <span style={{ fontSize: '11px', color: '#64748b' }}>
                  {inferenceStatus?.active_device ? `Device: ${inferenceStatus.active_device}` : 'CPU Fallback Ready'}
                </span>
              </div>
            </div>

            {/* Active Jobs & Audit Logs */}
            <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '30px' }}>
              <div className="glass-panel">
                <h4 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>Active Queued Tasks</h4>
                {!Array.isArray(jobs) || jobs.length === 0 ? (
                  <p style={{ color: '#64748b', fontSize: '14px' }}>No active background Celery tasks.</p>
                ) : (
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', textAlign: 'left' }}>
                        <th style={{ padding: '12px 8px', fontSize: '12px', color: '#94a3b8' }}>Job ID</th>
                        <th style={{ padding: '12px 8px', fontSize: '12px', color: '#94a3b8' }}>Type</th>
                        <th style={{ padding: '12px 8px', fontSize: '12px', color: '#94a3b8' }}>Status</th>
                        <th style={{ padding: '12px 8px', fontSize: '12px', color: '#94a3b8' }}>Created At</th>
                      </tr>
                    </thead>
                    <tbody>
                      {jobs.slice(0, 5).map(j => j && (
                        <tr key={j.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.02)' }}>
                          <td style={{ padding: '12px 8px', fontSize: '13px', fontFamily: 'monospace' }}>{(j.id || '').slice(0, 8)}</td>
                          <td style={{ padding: '12px 8px', fontSize: '13px' }}>{j.job_type}</td>
                          <td style={{ padding: '12px 8px' }}>
                            <span className={`badge ${j.status === 'Completed' ? 'success' : j.status === 'Running' ? 'info' : j.status === 'Failed' ? 'danger' : 'warning'}`}>
                              {j.status}
                            </span>
                          </td>
                          <td style={{ padding: '12px 8px', fontSize: '12px', color: '#64748b' }}>{new Date(j.created_at).toLocaleTimeString()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>

              <div className="glass-panel">
                <h4 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>GPU VRAM & Hardware Scheduler</h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', marginBottom: '6px' }}>
                      <span>GPU Core Utilization (Live)</span>
                      <span>{gpuStatus.gpu_util}%</span>
                    </div>
                    <div style={{ height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${gpuStatus.gpu_util}%`, background: 'var(--color-primary)', transition: 'width 0.3s' }}></div>
                    </div>
                  </div>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', marginBottom: '6px' }}>
                      <span>VRAM Allocation (RTX 4050 6GB)</span>
                      <span>{gpuStatus.vram_util}%</span>
                    </div>
                    <div style={{ height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${gpuStatus.vram_util}%`, background: 'var(--color-secondary)', transition: 'width 0.3s' }}></div>
                    </div>
                  </div>
                  <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.03)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: '#94a3b8' }}>
                      <AlertCircle size={14} className={inferenceStatus?.gpu_fallback_active ? 'warning' : 'success'} />
                      <span>
                        {inferenceStatus?.gpu_fallback_active 
                          ? "Dynamic CPU Inference fallback is currently active."
                          : "Inference running on CPU device."}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Datasets View */}
        {activeTab === 'datasets' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
            {activeMerge && (
              <div 
                className="glass-panel" 
                style={{ 
                  borderLeft: `4px solid ${
                    (activeMerge.status === 'completed' || activeMerge.status === 'ended') ? '#10b981' : 
                    activeMerge.status === 'failed' ? '#ef4444' : 
                    activeMerge.status === 'cancelled' ? '#f59e0b' : 
                    'var(--color-primary)'
                  }`, 
                  padding: '24px', 
                  background: 'rgba(30, 27, 75, 0.25)', 
                  width: '100%',
                  borderRadius: '12px',
                  boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
                  backdropFilter: 'blur(4px)',
                  border: '1px solid rgba(255, 255, 255, 0.05)',
                  borderLeftWidth: '4px'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '20px' }}>
                  <div>
                    <h5 style={{ fontSize: '16px', fontWeight: 700, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      {activeMerge.isCleaning ? "Batched Dataset Cleaning & Ingestion Pipeline" : 
                       (activeMerge.isMerged ? "Moses Parallel Merge & Ingestion Pipeline" : "Single File Ingestion & Validation Pipeline")}
                    </h5>
                    <p style={{ fontSize: '13px', color: '#94a3b8', marginTop: '4px' }}>
                      Dataset Name: <strong>"{activeMerge.name}"</strong>
                    </p>
                  </div>
                  <span 
                    className={`badge ${
                      (activeMerge.status === 'completed' || activeMerge.status === 'ended') ? 'success' : 
                      activeMerge.status === 'failed' ? 'danger' : 
                      activeMerge.status === 'cancelled' ? 'warning' : 'info'
                    }`} 
                    style={{ fontSize: '11px', padding: '4px 10px', textTransform: 'uppercase', fontWeight: 700 }}
                  >
                    {activeMerge.status}
                  </span>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: activeMerge.isMerged ? '1fr 1fr' : '1fr', gap: '20px', marginTop: '16px' }}>
                  
                  {activeMerge.isMerged && (
                    <div 
                      style={{ 
                        background: 'rgba(255, 255, 255, 0.02)', 
                        padding: '16px', 
                        borderRadius: '8px', 
                        border: '1px solid ' + (
                          activeMerge.phase === 'merging' ? 'rgba(139, 92, 246, 0.3)' : 
                          (activeMerge.phase === 'ingesting' || activeMerge.status === 'completed') ? 'rgba(16, 185, 129, 0.2)' : 'rgba(255, 255, 255, 0.05)'
                        )
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                        <span style={{ fontSize: '13px', fontWeight: 600, color: activeMerge.phase === 'merging' ? '#a78bfa' : '#94a3b8' }}>
                          Step 1: Merging separate files
                        </span>
                        <span style={{ fontSize: '11px', fontFamily: 'monospace', color: '#64748b' }}>
                          {activeMerge.phase === 'merging' ? 'In Progress' : 'Completed'}
                        </span>
                      </div>
                      
                      <div style={{ height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', overflow: 'hidden', margin: '10px 0' }}>
                        <div 
                          style={{ 
                            height: '100%', 
                            width: activeMerge.phase === 'merging' 
                              ? `${Math.min(100, (activeMerge.processedCount / (activeMerge.totalToProcess || 1)) * 100)}%` 
                              : '100%', 
                            background: 'linear-gradient(90deg, var(--color-primary) 0%, var(--color-secondary) 100%)',
                            transition: 'width 0.4s ease-out'
                          }}
                        ></div>
                      </div>

                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11.5px', color: '#64748b' }}>
                        <span>
                          {activeMerge.phase === 'merging' 
                            ? `${formatBytes(activeMerge.processedCount)} / ${formatBytes(activeMerge.totalToProcess)}`
                            : 'All bytes merged'
                          }
                        </span>
                        <span>
                          {activeMerge.linesMerged > 0 ? `${activeMerge.linesMerged.toLocaleString()} records` : ''}
                        </span>
                      </div>
                    </div>
                  )}

                  <div 
                    style={{ 
                      background: 'rgba(255, 255, 255, 0.02)', 
                      padding: '16px', 
                      borderRadius: '8px', 
                      border: '1px solid ' + (
                        (activeMerge.phase === 'ingesting' || activeMerge.phase === 'in progress' || activeMerge.phase === 'started') && activeMerge.status === 'processing' ? 'rgba(139, 92, 246, 0.3)' : 
                        (activeMerge.status === 'completed' || activeMerge.status === 'ended') ? 'rgba(16, 185, 129, 0.2)' : 'rgba(255, 255, 255, 0.05)'
                      )
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <span style={{ fontSize: '13px', fontWeight: 600, color: (activeMerge.phase === 'ingesting' || activeMerge.phase === 'in progress' || activeMerge.phase === 'started') ? '#a78bfa' : '#94a3b8' }}>
                        {activeMerge.isCleaning ? 'Cleaning & Processing Batch' : (activeMerge.isMerged ? 'Step 2: Ingesting & Validating' : 'Step 1: Ingesting & Validating')}
                      </span>
                      <span style={{ fontSize: '11px', fontFamily: 'monospace', color: '#64748b' }}>
                        {(activeMerge.status === 'completed' || activeMerge.status === 'ended') ? 'Completed' : 
                         (activeMerge.phase === 'ingesting' || activeMerge.phase === 'in progress' || activeMerge.phase === 'started') ? 'In Progress' : 'Pending'}
                      </span>
                    </div>

                    <div style={{ height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', overflow: 'hidden', margin: '10px 0' }}>
                      <div 
                        style={{ 
                          height: '100%', 
                          width: (activeMerge.status === 'completed' || activeMerge.status === 'ended') ? '100%' : 
                                 (activeMerge.phase === 'ingesting' || activeMerge.phase === 'in progress' || activeMerge.phase === 'started') 
                                 ? `${Math.min(100, (activeMerge.processedCount / (activeMerge.totalToProcess || 1)) * 100)}%` 
                                 : '0%', 
                          background: 'linear-gradient(90deg, #10b981 0%, #059669 100%)',
                          transition: 'width 0.4s ease-out'
                        }}
                      ></div>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11.5px', color: '#64748b' }}>
                      <span>
                        {activeMerge.isCleaning
                          ? `Cleaning: ${activeMerge.processedCount.toLocaleString()} / ${activeMerge.totalToProcess.toLocaleString()} records`
                          : (activeMerge.phase === 'ingesting'
                            ? `Validating: ${formatBytes(activeMerge.processedCount)} / ${formatBytes(activeMerge.totalToProcess)}`
                            : activeMerge.status === 'completed'
                            ? 'Validation completed'
                            : 'Awaiting merge completion...')
                        }
                      </span>
                      <span>
                        {(activeMerge.phase === 'ingesting' || activeMerge.phase === 'in progress' || activeMerge.phase === 'started') 
                          ? `${Math.min(100, (activeMerge.processedCount / (activeMerge.totalToProcess || 1)) * 100).toFixed(0)}%`
                          : ''
                        }
                      </span>
                    </div>
                  </div>

                </div>

                {activeMerge.status === 'failed' && (
                  <div style={{ marginTop: '16px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', padding: '12px', borderRadius: '6px', display: 'flex', alignItems: 'center', gap: '8px', color: '#fca5a5', fontSize: '12.5px' }}>
                    <AlertCircle size={16} />
                    <span><strong>Pipeline Failed:</strong> {activeMerge.error}</span>
                  </div>
                )}

                {activeMerge.status === 'cancelled' && (
                  <div style={{ marginTop: '16px', background: 'rgba(245, 158, 11, 0.1)', border: '1px solid rgba(245, 158, 11, 0.2)', padding: '12px', borderRadius: '6px', display: 'flex', alignItems: 'center', gap: '8px', color: '#fcd34d', fontSize: '12.5px' }}>
                    <AlertCircle size={16} />
                    <span><strong>Pipeline stopped:</strong> {activeMerge.error}</span>
                  </div>
                )}

                {(activeMerge.status === 'completed' || activeMerge.status === 'ended') && (
                  <div style={{ marginTop: '16px', background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.2)', padding: '12px', borderRadius: '6px', display: 'flex', alignItems: 'center', gap: '8px', color: '#a7f3d0', fontSize: '12.5px' }}>
                    <CheckCircle2 size={16} />
                    <span>
                      🎉 <strong>Pipeline Success!</strong> {activeMerge.isCleaning ? "The dataset was successfully cleaned and promoted to TrainReady." : "The dataset was successfully processed, validated, and registered."}
                    </span>
                  </div>
                )}

                {activeMerge.status === 'processing' && (
                  <button 
                    onClick={() => cancelPipeline(activeMerge.datasetId)} 
                    style={{ 
                      marginTop: '16px', 
                      background: 'rgba(239, 68, 68, 0.15)', 
                      border: '1px solid rgba(239, 68, 68, 0.3)', 
                      color: '#fca5a5',
                      padding: '8px 16px',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      fontSize: '13px',
                      fontWeight: 650,
                      boxShadow: 'none',
                      transition: 'all 0.2s'
                    }}
                    onMouseOver={e => { e.currentTarget.style.background = 'rgba(239, 68, 68, 0.3)'; e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.5)'; }}
                    onMouseOut={e => { e.currentTarget.style.background = 'rgba(239, 68, 68, 0.15)'; e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.3)'; }}
                  >
                    <AlertCircle size={16} /> Stop Pipeline
                  </button>
                )}

              </div>
            )}
            <div style={{ display: 'grid', gridTemplateColumns: showEngineInfo ? '0.4fr 0.6fr' : '1fr', gap: '30px', transition: 'all 0.3s ease' }}>
              {/* Ingestion form */}
              <div className="glass-panel" style={{ height: 'fit-content' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                  <h4 style={{ fontSize: '16px', fontWeight: 600, margin: 0 }}>Ingest Raw Dataset</h4>
                  <button 
                    type="button" 
                    className="secondary" 
                    style={{ padding: '2px 8px', fontSize: '11px', minWidth: 'auto', border: '1px solid rgba(255,255,255,0.05)', height: '24px' }}
                    onClick={() => setShowEngineInfo(!showEngineInfo)}
                  >
                    {showEngineInfo ? 'Hide Control Center' : 'Show Control Center'}
                  </button>
                </div>
                
                {/* Mode Selector Toggle */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '16px', padding: '2px', background: 'rgba(255,255,255,0.03)', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <button 
                    type="button"
                    onClick={() => setUploadMode('single')}
                    style={{ 
                      padding: '6px 12px', 
                      fontSize: '12px', 
                      borderRadius: '4px', 
                      border: 'none',
                      background: uploadMode === 'single' ? 'rgba(139, 92, 246, 0.2)' : 'transparent',
                      color: uploadMode === 'single' ? '#a78bfa' : '#94a3b8',
                      boxShadow: 'none',
                      cursor: 'pointer'
                    }}
                  >
                    Single Parallel File
                  </button>
                  <button 
                    type="button"
                    onClick={() => setUploadMode('moses')}
                    style={{ 
                      padding: '6px 12px', 
                      fontSize: '12px', 
                      borderRadius: '4px', 
                      border: 'none',
                      background: uploadMode === 'moses' ? 'rgba(139, 92, 246, 0.2)' : 'transparent',
                      color: uploadMode === 'moses' ? '#a78bfa' : '#94a3b8',
                      boxShadow: 'none',
                      cursor: 'pointer'
                    }}
                  >
                    Moses Parallel Files (Auto-Merge)
                  </button>
                </div>

                {/* Decision Guideline Banner */}
                <div style={{ background: 'rgba(255,255,255,0.02)', padding: '10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)', fontSize: '11.5px', color: '#94a3b8', marginBottom: '16px', lineHeight: '1.4' }}>
                  ℹ️ <strong>Which mode do I choose?</strong>
                  <ul style={{ margin: '4px 0 0 14px', padding: 0 }}>
                    <li>Choose <strong>Single Parallel File</strong> if your translation pairs are already in one unified file (e.g., CSV columns or tab-separated text).</li>
                    <li>Choose <strong>Moses Parallel Files</strong> if you have separate files for each language (e.g., raw <code>.en</code> and <code>.kn</code> text files).</li>
                  </ul>
                </div>

                <form onSubmit={handleUpload} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  <div>
                    <label style={{ fontSize: '12px', color: '#94a3b8', display: 'block', marginBottom: '6px' }}>Dataset Name</label>
                    <input type="text" value={datasetName} onChange={e => setDatasetName(e.target.value)} placeholder="e.g. Kannada translation validation" required />
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                    <div>
                      <label style={{ fontSize: '12px', color: '#94a3b8', display: 'block', marginBottom: '6px' }}>Source Language</label>
                      <select value={srcLang} onChange={e => setSrcLang(e.target.value)}>
                        <option value="en">English (en)</option>
                        <option value="kn">Kannada (kn)</option>
                        <option value="ml">Malayalam (ml)</option>
                      </select>
                    </div>
                    <div>
                      <label style={{ fontSize: '12px', color: '#94a3b8', display: 'block', marginBottom: '6px' }}>Target Language</label>
                      <select value={tgtLang} onChange={e => setTgtLang(e.target.value)}>
                        <option value="kn">Kannada (kn)</option>
                        <option value="ml">Malayalam (ml)</option>
                        <option value="en">English (en)</option>
                      </select>
                    </div>
                  </div>
                  
                  {uploadMode === 'single' ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      <div>
                        <label style={{ fontSize: '12px', color: '#94a3b8', display: 'block', marginBottom: '6px' }}>Select Unified File (CSV, TSV, JSONL, JSON, TXT)</label>
                        <input type="file" onChange={e => setSelectedFile(e.target.files[0])} required />
                      </div>
                      <div style={{ background: 'rgba(59, 130, 246, 0.05)', border: '1px dashed rgba(59, 130, 246, 0.2)', padding: '10px', borderRadius: '6px', fontSize: '11px', color: '#94a3b8', lineHeight: '1.4' }}>
                        💡 <strong>Supported Single File Formats:</strong>
                        <ul style={{ margin: '4px 0 0 16px', padding: 0 }}>
                          <li><strong>CSV / TSV:</strong> File with a header row (e.g. <code>src,tgt</code> or <code>source,target</code>) containing aligned pairs.</li>
                          <li><strong>JSON / JSONL:</strong> JSON lines or arrays containing items with key properties (e.g. <code>{"{src: '...', tgt: '...'}"}</code>).</li>
                          <li><strong>TXT:</strong> Simple text files containing tab-separated translations on each line (e.g. <code>SourceText \t TargetTranslation</code>).</li>
                        </ul>
                      </div>
                    </div>
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      <div>
                        <label style={{ fontSize: '12px', color: '#94a3b8', display: 'block', marginBottom: '6px' }}>Source Text File Path (on host or under data/)</label>
                        <input 
                          type="text" 
                          value={srcFilePath} 
                          onChange={e => setSrcFilePath(e.target.value)} 
                          placeholder="e.g. NLLB.en-kn.en" 
                          required 
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: '12px', color: '#94a3b8', display: 'block', marginBottom: '6px' }}>Target Text File Path (on host or under data/)</label>
                        <input 
                          type="text" 
                          value={tgtFilePath} 
                          onChange={e => setTgtFilePath(e.target.value)} 
                          placeholder="e.g. NLLB.en-kn.kn" 
                          required 
                        />
                      </div>
                      <div style={{ background: 'rgba(139, 92, 246, 0.05)', border: '1px dashed rgba(139, 92, 246, 0.2)', padding: '10px', borderRadius: '6px', fontSize: '11px', color: '#94a3b8', lineHeight: '1.4' }}>
                        💡 <strong>Aesthetic MLOps Tip:</strong> For massive datasets like NLLB (34M sentences), copy files directly into <code>D:\AI Translator\data\raw\</code> and specify their filenames above. This merges them instantly in the backend without slow uploads!
                      </div>
                    </div>
                  )}

                  <button type="submit" style={{ marginTop: '8px' }}>
                    <ArrowRight size={16} /> {uploadMode === 'single' ? 'Upload & Ingest v1' : 'Auto-Merge & Ingest v1'}
                  </button>
                </form>
              </div>
              
              {/* MLOps Control & Monitor Center */}
              {showEngineInfo && (
                <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: '450px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '10px' }}>
                    <h4 style={{ fontSize: '16px', fontWeight: 700, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '6px', margin: 0 }}>
                      <Sparkles size={16} color="var(--color-primary)" /> MLOps Control Center
                    </h4>
                    <span className="badge info" style={{ fontSize: '10px' }}>ONLINE</span>
                  </div>

                  {/* Tabs Selector */}
                  <div style={{ display: 'flex', gap: '4px', marginBottom: '16px', padding: '2px', background: 'rgba(255,255,255,0.03)', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <button 
                      type="button"
                      style={{ 
                        flex: 1, padding: '6px 8px', fontSize: '11px', borderRadius: '4px', border: 'none', height: '28px', minWidth: 'auto',
                        background: activeConsoleTab === 'ledger' ? 'linear-gradient(135deg, #06b6d4 0%, #0891b2 100%)' : 'transparent',
                        color: activeConsoleTab === 'ledger' ? '#ffffff' : '#94a3b8',
                        cursor: 'pointer', fontWeight: 600, transition: 'all 0.2s'
                      }}
                      onClick={() => setActiveConsoleTab('ledger')}
                    >
                      Version Ledger
                    </button>
                    <button 
                      type="button"
                      style={{ 
                        flex: 1, padding: '6px 8px', fontSize: '11px', borderRadius: '4px', border: 'none', height: '28px', minWidth: 'auto',
                        background: activeConsoleTab === 'celery' ? 'linear-gradient(135deg, #06b6d4 0%, #0891b2 100%)' : 'transparent',
                        color: activeConsoleTab === 'celery' ? '#ffffff' : '#94a3b8',
                        cursor: 'pointer', fontWeight: 600, transition: 'all 0.2s'
                      }}
                      onClick={() => setActiveConsoleTab('celery')}
                    >
                      Celery Monitor
                    </button>
                    <button 
                      type="button"
                      style={{ 
                        flex: 1, padding: '6px 8px', fontSize: '11px', borderRadius: '4px', border: 'none', height: '28px', minWidth: 'auto',
                        background: activeConsoleTab === 'hardware' ? 'linear-gradient(135deg, #06b6d4 0%, #0891b2 100%)' : 'transparent',
                        color: activeConsoleTab === 'hardware' ? '#ffffff' : '#94a3b8',
                        cursor: 'pointer', fontWeight: 600, transition: 'all 0.2s'
                      }}
                      onClick={() => setActiveConsoleTab('hardware')}
                    >
                      Hardware
                    </button>
                  </div>

                  {/* Tab Body */}
                  <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
                    {activeConsoleTab === 'ledger' && (() => {
                      const ledgerCleaned = safeDatasets.filter(d => d && d.version && (d.version.includes("cleaned") || ["Processed", "TrainReady", "TrainingUsed"].includes(d.status)));
                      const ledgerBatches = safeDatasets.filter(d => d && d.version && !ledgerCleaned.some(c => c.id === d.id) && d.version.includes("subset"));
                      const ledgerParents = safeDatasets.filter(d => d && d.version && !ledgerCleaned.some(c => c.id === d.id) && !ledgerBatches.some(b => b.id === d.id));

                      const renderSubLedger = (title, list, color) => (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '16px' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '6px' }}>
                            <h6 style={{ fontSize: '12px', fontWeight: 700, color: color, display: 'flex', alignItems: 'center', gap: '6px', margin: 0 }}>
                              {title} <span style={{ fontSize: '10px', color: '#64748b', fontWeight: 400 }}>({list.length})</span>
                            </h6>
                          </div>
                          {list.length === 0 ? (
                            <p style={{ color: '#64748b', fontSize: '11px', paddingLeft: '4px', margin: '4px 0 10px 0' }}>No versions registered in this phase.</p>
                          ) : (
                            <div style={{ overflowX: 'auto', marginBottom: '8px' }}>
                              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11.5px', textAlign: 'left' }}>
                                <thead>
                                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.03)', color: '#64748b' }}>
                                    <th style={{ padding: '4px 4px', width: '30px' }}>#</th>
                                    <th style={{ padding: '4px 4px', width: '35%' }}>Version</th>
                                    <th style={{ padding: '4px 4px', width: '15%' }}>Lang</th>
                                    <th style={{ padding: '4px 4px', width: '20%' }}>Records</th>
                                    <th style={{ padding: '4px 4px', width: '15%' }}>Status</th>
                                    <th style={{ padding: '4px 4px', textAlign: 'right', width: '15%' }}>Actions</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {list.map(d => {
                                    const globalIdx = safeDatasets.findIndex(x => x.id === d.id) + 1;
                                    return (
                                      <tr key={d.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.01)', verticalAlign: 'middle' }}>
                                        <td style={{ padding: '6px 4px', color: '#64748b', fontWeight: 650 }}>
                                          {globalIdx}
                                        </td>
                                        <td style={{ padding: '6px 4px', fontWeight: 600, color: '#e2e8f0', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '140px' }} title={d.version}>
                                          {d.version}
                                        </td>
                                        <td style={{ padding: '6px 4px', whiteSpace: 'nowrap' }}>{d.src_lang}➔{d.tgt_lang}</td>
                                        <td style={{ padding: '6px 4px' }}>{d.record_count?.toLocaleString() || 0}</td>
                                        <td style={{ padding: '6px 4px' }}>
                                          <span className={`badge ${
                                            d.status === 'Validated' || d.status === 'Processed' || d.status === 'TrainReady' ? 'success' :
                                            d.status === 'Processing' ? 'warning' :
                                            d.status === 'Failed' ? 'danger' : 'muted'
                                          }`} style={{ fontSize: '9px', padding: '1px 4px' }}>
                                            {d.status}
                                          </span>
                                        </td>
                                        <td style={{ padding: '6px 4px', textAlign: 'right', whiteSpace: 'nowrap' }}>
                                          <button 
                                            className="secondary" 
                                            style={{ padding: '2px 4px', fontSize: '9px', minWidth: 'auto', marginRight: '4px', height: '20px' }}
                                            onClick={() => fetchPreview(d.id)}
                                            disabled={d.status === 'Processing'}
                                          >
                                            View
                                          </button>
                                          <button 
                                            className="secondary" 
                                            style={{ padding: '2px 4px', fontSize: '9px', minWidth: 'auto', height: '20px', background: 'rgba(239, 68, 68, 0.05)', color: '#fca5a5', border: '1px solid rgba(239, 68, 68, 0.1)' }}
                                            onClick={() => handleDelete(d.id)}
                                          >
                                            Del
                                          </button>
                                        </td>
                                      </tr>
                                    );
                                  })}
                                </tbody>
                              </table>
                            </div>
                          )}
                        </div>
                      );

                      return (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#64748b', paddingBottom: '4px', borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                            <span>Unified Dataset Registry</span>
                            <span>Total: {safeDatasets.length}</span>
                          </div>
                          {safeDatasets.length === 0 ? (
                            <p style={{ color: '#64748b', fontSize: '12px', textAlign: 'center', marginTop: '20px' }}>No dataset versions registered.</p>
                          ) : (
                            <div style={{ display: 'flex', flexDirection: 'column' }}>
                              {renderSubLedger("Phase 1: Parent Corpora (Raw & Merged)", ledgerParents, "#a78bfa")}
                              {renderSubLedger("Phase 2: Batched Subsets", ledgerBatches, "#22d3ee")}
                              {renderSubLedger("Phase 3: Cleaned & Train-Ready", ledgerCleaned, "#34d399")}
                            </div>
                          )}
                        </div>
                      );
                    })()}

                    {activeConsoleTab === 'celery' && (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#64748b', paddingBottom: '4px', borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                          <span>Celery Task Queue</span>
                          <span>Active: {jobs.filter(j => j && j.status === 'Running').length}</span>
                        </div>
                        {!Array.isArray(jobs) || jobs.length === 0 ? (
                          <p style={{ color: '#64748b', fontSize: '12px', textAlign: 'center', marginTop: '20px' }}>No active or completed background tasks.</p>
                        ) : (
                          <div style={{ overflowX: 'auto' }}>
                            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11.5px', textAlign: 'left' }}>
                              <thead>
                                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', color: '#94a3b8' }}>
                                  <th style={{ padding: '6px 4px' }}>Job ID</th>
                                  <th style={{ padding: '6px 4px' }}>Type</th>
                                  <th style={{ padding: '6px 4px' }}>Status</th>
                                  <th style={{ padding: '6px 4px' }}>Time</th>
                                  <th style={{ padding: '6px 4px', textAlign: 'right' }}>Actions</th>
                                </tr>
                              </thead>
                              <tbody>
                                {jobs.map(j => j && (
                                  <tr key={j.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.02)' }}>
                                    <td style={{ padding: '8px 4px', fontFamily: 'monospace', color: '#38bdf8' }}>
                                      {(j.id || '').slice(0, 8)}
                                    </td>
                                    <td style={{ padding: '8px 4px', textTransform: 'capitalize' }}>
                                      {j.job_type || 'Unknown'}
                                    </td>
                                    <td style={{ padding: '8px 4px' }}>
                                      <span className={`badge ${
                                        j.status === 'Completed' ? 'success' :
                                        j.status === 'Running' ? 'info' :
                                        j.status === 'Failed' ? 'danger' : 'warning'
                                      }`} style={{ fontSize: '9px', padding: '1px 4px' }}>
                                        {j.status}
                                      </span>
                                    </td>
                                    <td style={{ padding: '8px 4px', color: '#64748b' }}>
                                      {j.created_at ? new Date(j.created_at).toLocaleTimeString() : 'N/A'}
                                    </td>
                                    <td style={{ padding: '8px 4px', textAlign: 'right' }}>
                                      {(j.status === 'Running' || j.status === 'Starting' || j.status === 'Queued' || j.status === 'Paused') && (
                                        <button 
                                          style={{ 
                                            background: 'rgba(239, 68, 68, 0.1)', 
                                            color: '#fca5a5', 
                                            border: '1px solid rgba(239, 68, 68, 0.2)', 
                                            padding: '2px 6px', 
                                            fontSize: '10px', 
                                            borderRadius: '4px',
                                            cursor: 'pointer',
                                            boxShadow: 'none',
                                            width: 'auto',
                                            margin: 0
                                          }}
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            const confirmMsg = j.status === 'Queued' 
                                              ? "Are you sure you want to remove this job from the queue?" 
                                              : "Are you sure you want to cancel this job?";
                                            if (window.confirm(confirmMsg)) {
                                              cancelJob(j.id);
                                            }
                                          }}
                                        >
                                          {j.status === 'Queued' ? 'Remove' : 'Cancel'}
                                        </button>
                                      )}
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}
                      </div>
                    )}

                    {activeConsoleTab === 'hardware' && (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginTop: '6px' }}>
                        <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '6px' }}>
                            <span style={{ color: '#94a3b8' }}>CPU Core Utilization (Live)</span>
                            <span style={{ fontWeight: 600 }}>{gpuStatus.cpu_util}%</span>
                          </div>
                          <div style={{ height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', overflow: 'hidden' }}>
                            <div style={{ height: '100%', width: `${gpuStatus.cpu_util}%`, background: 'linear-gradient(90deg, #06b6d4 0%, #0891b2 100%)', transition: 'width 0.5s ease-out' }}></div>
                          </div>
                        </div>

                        <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '6px' }}>
                            <span style={{ color: '#94a3b8' }}>GPU VRAM Allocation (Live)</span>
                            <span style={{ fontWeight: 600 }}>{gpuStatus.vram_util}%</span>
                          </div>
                          <div style={{ height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', overflow: 'hidden' }}>
                            <div style={{ height: '100%', width: `${gpuStatus.vram_util}%`, background: 'linear-gradient(90deg, #a78bfa 0%, #8b5cf6 100%)', transition: 'width 0.5s ease-out' }}></div>
                          </div>
                        </div>

                        <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.04)', fontSize: '11px', color: '#94a3b8', lineHeight: '1.5' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px', color: '#e2e8f0', fontWeight: 600 }}>
                            <span style={{ display: 'inline-block', width: '6px', height: '6px', borderRadius: '50%', background: '#10b981' }}></span>
                            RTX 4050 Scheduler Status
                          </div>
                          <div>Active Job ID: <code style={{ color: '#a78bfa' }}>{gpuStatus.active_job_id || 'None (Idle)'}</code></div>
                          <div>Celery Worker Queue status: <strong style={{ color: '#34d399' }}>Sequentially Active</strong></div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Interactive MLOps Pipeline Flow Visualizer */}
            <div className="glass-panel" style={{ padding: '16px', background: 'rgba(15, 23, 42, 0.4)', borderColor: 'rgba(255, 255, 255, 0.05)', borderRadius: '16px', marginBottom: '10px' }}>
              <div 
                style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
                onClick={() => setShowFlowVisualizer(!showFlowVisualizer)}
              >
                <h4 style={{ fontSize: '13px', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '1px', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <RefreshCw size={14} style={{ animation: 'spin 4s linear infinite' }} /> Real-time MLOps Pipeline Flow
                </h4>
                <button 
                  className="secondary" 
                  style={{ padding: '2px 8px', fontSize: '11px', minWidth: 'auto', border: '1px solid rgba(255,255,255,0.05)', height: '24px' }}
                >
                  {showFlowVisualizer ? 'Hide Visualizer' : 'Show Visualizer'}
                </button>
              </div>
              
              {showFlowVisualizer && (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px', marginTop: '16px' }}>
                  {/* Stage 1 */}
                  <div style={{ flex: 1, minWidth: '160px', padding: '12px', background: 'rgba(139, 92, 246, 0.05)', border: '1px solid rgba(139, 92, 246, 0.15)', borderRadius: '8px', textAlign: 'center' }}>
                    <div style={{ fontSize: '11px', color: '#a78bfa', fontWeight: 600 }}>1. INGEST & MERGE</div>
                    <div style={{ fontSize: '20px', fontWeight: 800, margin: '4px 0', color: '#f8fafc' }}>
                      {safeDatasets.filter(d => d && d.version && !d.version.includes("subset") && !d.version.includes("cleaned") && d.status !== 'Archived').length}
                    </div>
                    <div style={{ fontSize: '10px', color: '#64748b' }}>Raw / Merged Corpora</div>
                  </div>

                  <div style={{ color: '#64748b', fontSize: '18px', fontWeight: 700 }}>➔</div>

                  {/* Stage 2 */}
                  <div style={{ flex: 1, minWidth: '160px', padding: '12px', background: 'rgba(6, 182, 212, 0.05)', border: '1px solid rgba(6, 182, 212, 0.15)', borderRadius: '8px', textAlign: 'center' }}>
                    <div style={{ fontSize: '11px', color: '#22d3ee', fontWeight: 600 }}>2. SENTENCE BATCHING</div>
                    <div style={{ fontSize: '20px', fontWeight: 800, margin: '4px 0', color: '#f8fafc' }}>
                      {safeDatasets.filter(d => d && d.version && d.version.includes("subset") && !d.version.includes("cleaned") && d.status !== 'Archived').length}
                    </div>
                    <div style={{ fontSize: '10px', color: '#64748b' }}>Batched Subsets</div>
                  </div>

                  <div style={{ color: '#64748b', fontSize: '18px', fontWeight: 700 }}>➔</div>

                  {/* Stage 3 */}
                  <div style={{ flex: 1, minWidth: '160px', padding: '12px', background: 'rgba(16, 185, 129, 0.05)', border: '1px solid rgba(16, 185, 129, 0.15)', borderRadius: '8px', textAlign: 'center' }}>
                    <div style={{ fontSize: '11px', color: '#34d399', fontWeight: 600 }}>3. QUALITY CLEANING</div>
                    <div style={{ fontSize: '20px', fontWeight: 800, margin: '4px 0', color: '#f8fafc' }}>
                      {safeDatasets.filter(d => d && d.version && (d.version.includes("cleaned") || ["TrainReady", "TrainingUsed"].includes(d.status)) && d.status !== 'Archived').length}
                    </div>
                    <div style={{ fontSize: '10px', color: '#64748b' }}>Clean Train-Ready Sets</div>
                  </div>

                  <div style={{ color: '#64748b', fontSize: '18px', fontWeight: 700 }}>➔</div>

                  {/* Stage 4 */}
                  <div style={{ flex: 1, minWidth: '160px', padding: '12px', background: gpuStatus.is_locked ? 'rgba(245, 158, 11, 0.05)' : 'rgba(255, 255, 255, 0.01)', border: '1px solid ' + (gpuStatus.is_locked ? 'rgba(245, 158, 11, 0.2)' : 'rgba(255, 255, 255, 0.05)'), borderRadius: '8px', textAlign: 'center' }}>
                    <div style={{ fontSize: '11px', color: gpuStatus.is_locked ? '#fcd34d' : '#94a3b8', fontWeight: 600 }}>4. MODEL TRAINING</div>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', margin: '4px 0' }}>
                      <span style={{ 
                        display: 'inline-block', 
                        width: '8px', 
                        height: '8px', 
                        borderRadius: '50%', 
                        background: gpuStatus.is_locked ? '#f59e0b' : '#10b981',
                        animation: gpuStatus.is_locked ? 'pulseGlow 1.5s infinite' : 'none'
                      }}></span>
                      <span style={{ fontSize: '14px', fontWeight: 800, color: '#f8fafc' }}>
                        {gpuStatus.is_locked ? 'Active' : 'Idle'}
                      </span>
                    </div>
                    <div style={{ fontSize: '10px', color: '#64748b' }}>RTX 4050 GPU Scheduler</div>
                  </div>

                  <div style={{ color: '#64748b', fontSize: '18px', fontWeight: 700 }}>➔</div>

                  {/* Stage 5 */}
                  <div style={{ flex: 1, minWidth: '160px', padding: '12px', background: inferenceStatus?.model_version ? 'rgba(16, 185, 129, 0.05)' : 'rgba(255, 255, 255, 0.01)', border: '1px solid ' + (inferenceStatus?.model_version ? 'rgba(16, 185, 129, 0.2)' : 'rgba(255, 255, 255, 0.05)'), borderRadius: '8px', textAlign: 'center' }}>
                    <div style={{ fontSize: '11px', color: inferenceStatus?.model_version ? '#34d399' : '#94a3b8', fontWeight: 600 }}>5. MODEL SERVING</div>
                    <div style={{ fontSize: '13px', fontWeight: 800, margin: '4px 0', color: '#f8fafc', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={inferenceStatus?.model_version || 'Offline'}>
                      {inferenceStatus?.model_version ? `Served: ${inferenceStatus.model_version}` : 'CPU Fallback'}
                    </div>
                    <div style={{ fontSize: '10px', color: '#64748b' }}>Stateless Inference</div>
                  </div>
                </div>
              )}
            </div>

            {/* Developer Control Centre: Data Pipeline Lifecycle */}
            {(() => {
              const parentCorpora = safeDatasets.filter(d => d && d.version && !d.version.includes("subset") && !d.version.includes("cleaned") && d.status !== 'Archived');
              const batchedSubsets = safeDatasets.filter(d => d && d.version && d.version.includes("subset") && !d.version.includes("cleaned") && d.status !== 'Archived');
              
              const cleanTrainReady = safeDatasets.filter(d => d && d.version && (d.version.includes("cleaned") || ["TrainReady", "TrainingUsed"].includes(d.status)));
              const displayedCleanReady = cleanTrainReady.filter(d => d && (d.status !== 'Archived' || showArchived));
              
              return (
                <div style={{ marginTop: '16px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '20px' }}>
                    <Sparkles size={20} color="var(--color-primary)" />
                    <h4 style={{ fontSize: '18px', fontWeight: 700 }}>Data Pipeline Lifecycle Control Centre</h4>
                  </div>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '24px' }}>
                    {/* COLUMN 1: Parent Corpora (Raw & Merged) */}
                    <div className="glass-panel" style={{ background: 'rgba(10, 15, 30, 0.3)', minHeight: '400px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '10px' }}>
                        <h5 style={{ fontSize: '14px', fontWeight: 700, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '6px' }}>
                          Phase 1: Parent Corpora <span style={{ fontSize: '12px', color: '#64748b', fontWeight: 400 }}>({parentCorpora.length})</span>
                        </h5>
                        <span style={{ fontSize: '10px', color: '#a78bfa', background: 'rgba(139, 92, 246, 0.1)', padding: '2px 6px', borderRadius: '4px', border: '1px solid rgba(139, 92, 246, 0.2)', fontWeight: 600 }}>RAW / MERGED</span>
                      </div>
                      
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', flex: 1, maxHeight: '600px', overflowY: 'auto', paddingRight: '4px' }}>
                        {parentCorpora.map(ver => (
                          <ParentCard 
                            key={ver.id}
                            ver={ver} 
                            activeMerge={activeMerge}
                            fetchPreview={fetchPreview} 
                            handleDelete={handleDelete} 
                            handleSubsetSubmit={handleSubsetSubmit} 
                            index={safeDatasets.findIndex(d => d.id === ver.id) + 1}
                          />
                        ))}
                        {parentCorpora.length === 0 && renderEmptyColumn("No parent corpora registered. Ingest a raw or merged Moses parallel dataset.")}
                      </div>
                    </div>

                    {/* COLUMN 2: Batched Subsets */}
                    <div className="glass-panel" style={{ background: 'rgba(10, 15, 30, 0.3)', minHeight: '400px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '10px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <h5 style={{ fontSize: '14px', fontWeight: 700, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '6px', margin: 0 }}>
                            Phase 2: Batched Subsets <span style={{ fontSize: '12px', color: '#64748b', fontWeight: 400 }}>({batchedSubsets.length})</span>
                          </h5>
                          <span style={{ fontSize: '10px', color: '#22d3ee', background: 'rgba(6, 182, 212, 0.1)', padding: '2px 6px', borderRadius: '4px', border: '1px solid rgba(6, 182, 212, 0.2)', fontWeight: 600 }}>BATCHED</span>
                        </div>
                        {batchedSubsets.length > 0 && (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '4px' }}>
                            <div style={{ display: 'flex', gap: '8px' }}>
                              <button 
                                className="secondary" 
                                style={{ 
                                  padding: '2px 8px', 
                                  fontSize: '10px', 
                                  background: selectedBatches.length > 0 ? 'rgba(6, 182, 212, 0.15)' : 'transparent', 
                                  color: selectedBatches.length > 0 ? '#22d3ee' : '#64748b', 
                                  border: '1px solid ' + (selectedBatches.length > 0 ? 'rgba(6, 182, 212, 0.3)' : 'rgba(255,255,255,0.05)'),
                                  height: '24px',
                                  minWidth: 'auto',
                                  flex: 1
                                }} 
                                onClick={() => handleBulkClean(selectedBatches)}
                                disabled={selectedBatches.length === 0}
                              >
                                Clean Selected ({selectedBatches.length})
                              </button>
                            </div>
                            <div style={{ display: 'flex', gap: '8px' }}>
                              <button 
                                className="secondary" 
                                style={{ 
                                  padding: '2px 8px', 
                                  fontSize: '10px', 
                                  background: selectedBatches.length > 0 ? 'rgba(239, 68, 68, 0.15)' : 'transparent', 
                                  color: selectedBatches.length > 0 ? '#fca5a5' : '#64748b', 
                                  border: '1px solid ' + (selectedBatches.length > 0 ? 'rgba(239, 68, 68, 0.3)' : 'rgba(255,255,255,0.05)'),
                                  height: '24px',
                                  minWidth: 'auto',
                                  flex: 1
                                }} 
                                onClick={() => handleBulkDelete(selectedBatches, "selected batched subsets")}
                                disabled={selectedBatches.length === 0}
                              >
                                Delete Selected ({selectedBatches.length})
                              </button>
                              <button 
                                className="secondary" 
                                style={{ 
                                  padding: '2px 8px', 
                                  fontSize: '10px', 
                                  background: 'rgba(239, 68, 68, 0.05)', 
                                  color: '#fca5a5', 
                                  border: '1px solid rgba(239, 68, 68, 0.1)',
                                  height: '24px',
                                  minWidth: 'auto',
                                  flex: 1
                                }} 
                                onClick={() => handleBulkDelete(batchedSubsets.map(d => d.id), "all batched subsets")}
                              >
                                Delete All Batched
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                      
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', flex: 1, maxHeight: '600px', overflowY: 'auto', paddingRight: '4px' }}>
                        {batchedSubsets.map(ver => (
                          <BatchedCard 
                            key={ver.id}
                            ver={ver} 
                            activeMerge={activeMerge} 
                            fetchPreview={fetchPreview} 
                            handleDelete={handleDelete} 
                            triggerProcessing={triggerProcessing} 
                            cancelPipeline={cancelPipeline} 
                            cancelJob={cancelJob}
                            isSelected={selectedBatches.includes(ver.id)}
                            onSelectToggle={() => {
                              setSelectedBatches(prev => 
                                prev.includes(ver.id) ? prev.filter(id => id !== ver.id) : [...prev, ver.id]
                              );
                            }}
                            jobs={jobs}
                            index={safeDatasets.findIndex(d => d.id === ver.id) + 1}
                          />
                        ))}
                        {batchedSubsets.length === 0 && renderEmptyColumn("No subsets generated. Generate a smaller batch from a parent corpus above.")}
                      </div>
                    </div>

                    {/* COLUMN 3: Cleaned & Train-Ready */}
                    <div className="glass-panel" style={{ background: 'rgba(10, 15, 30, 0.3)', minHeight: '400px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '10px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <h5 style={{ fontSize: '14px', fontWeight: 700, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '6px', margin: 0 }}>
                            Phase 3: Cleaned & Train-Ready <span style={{ fontSize: '12px', color: '#64748b', fontWeight: 400 }}>({displayedCleanReady.length})</span>
                          </h5>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <label style={{ fontSize: '10px', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', margin: 0 }}>
                              <input 
                                type="checkbox" 
                                checked={showArchived} 
                                onChange={(e) => setShowArchived(e.target.checked)} 
                                style={{ width: 'auto', margin: 0, padding: 0, height: 'auto' }} 
                              />
                              Show Archived
                            </label>
                            <span style={{ fontSize: '10px', color: '#34d399', background: 'rgba(16, 185, 129, 0.1)', padding: '2px 6px', borderRadius: '4px', border: '1px solid rgba(16, 185, 129, 0.2)', fontWeight: 600 }}>CLEANED</span>
                          </div>
                        </div>
                        {displayedCleanReady.length > 0 && (
                          <div style={{ display: 'flex', gap: '8px', marginTop: '4px' }}>
                            <button 
                              className="secondary" 
                              style={{ 
                                padding: '2px 8px', 
                                fontSize: '10px', 
                                background: selectedCleaned.length > 0 ? 'rgba(239, 68, 68, 0.15)' : 'transparent', 
                                color: selectedCleaned.length > 0 ? '#fca5a5' : '#64748b', 
                                border: '1px solid ' + (selectedCleaned.length > 0 ? 'rgba(239, 68, 68, 0.3)' : 'rgba(255,255,255,0.05)'),
                                height: '24px',
                                minWidth: 'auto',
                                flex: 1
                              }} 
                              onClick={() => handleBulkDelete(selectedCleaned, "selected cleaned datasets")}
                              disabled={selectedCleaned.length === 0}
                            >
                              Delete Selected ({selectedCleaned.length})
                            </button>
                            <button 
                              className="secondary" 
                              style={{ 
                                padding: '2px 8px', 
                                fontSize: '10px', 
                                background: 'rgba(239, 68, 68, 0.05)', 
                                color: '#fca5a5', 
                                border: '1px solid rgba(239, 68, 68, 0.1)',
                                height: '24px',
                                minWidth: 'auto',
                                flex: 1
                              }} 
                              onClick={() => handleBulkDelete(displayedCleanReady.map(d => d.id), "all cleaned datasets")}
                            >
                              Delete All Cleaned
                            </button>
                          </div>
                        )}
                      </div>
                      
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', flex: 1, maxHeight: '600px', overflowY: 'auto', paddingRight: '4px' }}>
                        {displayedCleanReady.map(ver => (
                          <CleanedCard 
                            key={ver.id}
                            ver={ver} 
                            experiments={experiments} 
                            jobs={jobs} 
                            models={models} 
                            inferenceStatus={inferenceStatus} 
                            fetchPreview={fetchPreview} 
                            handleLoadInTrainer={handleLoadInTrainer} 
                            handleArchive={handleArchive} 
                            handleDelete={handleDelete} 
                            deployModel={deployModel} 
                            isSelected={selectedCleaned.includes(ver.id)}
                            onSelectToggle={() => {
                              setSelectedCleaned(prev => 
                                prev.includes(ver.id) ? prev.filter(id => id !== ver.id) : [...prev, ver.id]
                              );
                            }}
                            index={safeDatasets.findIndex(d => d.id === ver.id) + 1}
                            cancelJob={cancelJob}
                            resetFailedJob={resetFailedJob}
                            pauseJob={pauseJob}
                            resumeJob={resumeJob}
                          />
                        ))}
                        {displayedCleanReady.length === 0 && renderEmptyColumn("No cleaned data ready. Run the cleaning pipeline on any batched subset.")}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })()}

            {/* Preview Records Section */}
            {previewData && (
              <div className="glass-panel" style={{ position: 'relative' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                  <h4 style={{ fontSize: '16px', fontWeight: 600, margin: 0 }}>Data Preview: {selectedVersionForPreview}</h4>
                  <button 
                    className="secondary" 
                    style={{ padding: '4px 8px', fontSize: '12px', minWidth: 'auto', border: '1px solid rgba(255,255,255,0.1)' }} 
                    onClick={() => {
                      setPreviewData(null);
                      setSelectedVersionForPreview(null);
                    }}
                  >
                    Close Preview
                  </button>
                </div>
                {!Array.isArray(previewData) || previewData.length === 0 ? (
                  <p style={{ color: '#64748b' }}>No record preview available for this version.</p>
                ) : (
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', textAlign: 'left' }}>
                        <th style={{ padding: '12px', fontSize: '13px', color: '#94a3b8', width: '50%' }}>Source (English)</th>
                        <th style={{ padding: '12px', fontSize: '13px', color: '#94a3b8', width: '50%' }}>Target Translation</th>
                      </tr>
                    </thead>
                    <tbody>
                      {/* Render mock rows if mock version */}
                      {selectedVersionForPreview === 'mock-dataset-v2' ? (
                        <>
                          <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.02)' }}>
                            <td style={{ padding: '12px', fontSize: '14px' }}>What is your name?</td>
                            <td style={{ padding: '12px', fontSize: '14px', color: 'var(--color-secondary)' }}>ನಿಮ್ಮ ಹೆಸರು ಏನು?</td>
                          </tr>
                          <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.02)' }}>
                            <td style={{ padding: '12px', fontSize: '14px' }}>Where is the library?</td>
                            <td style={{ padding: '12px', fontSize: '14px', color: 'var(--color-secondary)' }}>ಗ್ರಂಥಾಲಯ ಎಲ್ಲಿದೆ?</td>
                          </tr>
                          <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.02)' }}>
                            <td style={{ padding: '12px', fontSize: '14px' }}>Thank you very much.</td>
                            <td style={{ padding: '12px', fontSize: '14px', color: 'var(--color-secondary)' }}>ತುಂಬಾ ಧನ್ಯವಾದಗಳು.</td>
                          </tr>
                        </>
                      ) : (
                        previewData.map((r, i) => r && (
                          <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.02)' }}>
                            <td style={{ padding: '12px', fontSize: '14px' }}>{r.src}</td>
                            <td style={{ padding: '12px', fontSize: '14px', color: 'var(--color-secondary)' }}>{r.tgt}</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                )}
              </div>
            )}
          </div>
        )}

        {/* Training View */}
        {activeTab === 'training' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
            {activeTrainingJob && (
              <div className="glass-panel" style={{ borderLeft: '4px solid var(--color-primary)', background: 'rgba(139, 92, 246, 0.02)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <div className="pulse-dot" style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#8b5cf6', boxShadow: '0 0 8px #8b5cf6' }}></div>
                    <h4 style={{ fontSize: '16px', fontWeight: 600, margin: 0 }}>Active Training Job: <span style={{ fontFamily: 'monospace', color: '#a78bfa' }}>{activeTrainingJob.id.slice(0, 8)}</span></h4>
                  </div>
                  <span className="badge warning" style={{ textTransform: 'uppercase', letterSpacing: '0.05em', fontSize: '10px' }}>
                    {activeTrainingJob.status}
                  </span>
                </div>

                <TrainingProgressPanel 
                  progress={activeTrainingJob.config?.progress} 
                  jobStatus={activeTrainingJob.status} 
                  errorLog={activeTrainingJob.error_log} 
                  jobId={activeTrainingJob.id}
                  onCancel={cancelJob}
                  onPause={pauseJob}
                  onResume={resumeJob}
                />
              </div>
            )}
            <div style={{ display: 'grid', gridTemplateColumns: '0.8fr 1.2fr', gap: '30px' }}>
              {/* Submission Form */}
              <div className="glass-panel" style={{ height: 'fit-content' }}>
                <h4 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '20px' }}>Configure Training Hyperparameters</h4>
                <form onSubmit={handleTrainSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  <div>
                    <label style={{ fontSize: '12px', color: '#94a3b8', display: 'block', marginBottom: '6px' }}>Dataset Version</label>
                    <select value={selectedDatasetId} onChange={e => setSelectedDatasetId(e.target.value)} required>
                      <option value="">-- Select TrainReady Version --</option>
                      {safeDatasets.filter(d => d && ["Processed", "TrainReady", "TrainingUsed"].includes(d.status)).map(d => {
                        const typeTag = d.processing_history?.version_type ? `[${d.processing_history.version_type.toUpperCase()}] ` : '';
                        const globalIdx = safeDatasets.findIndex(x => x.id === d.id) + 1;
                        return (
                          <option key={d.id} value={d.id}>
                            #{globalIdx} {typeTag}{d.version} ({d.record_count.toLocaleString()} pairs)
                          </option>
                        );
                      })}
                      {safeDatasets.length === 0 && (
                        <option value="mock-dataset-v4">Kannada corpus train ready (v2_cleaned) [Demo]</option>
                      )}
                    </select>
                  </div>
                  <div>
                    <label style={{ fontSize: '12px', color: '#94a3b8', display: 'block', marginBottom: '6px' }}>Base Model / Starting Checkpoint</label>
                    <select value={trainConfig.model_name} onChange={e => setTrainConfig({...trainConfig, model_name: e.target.value})}>
                      <optgroup label="Standard Pre-trained Models">
                        <option value="facebook/mbart-large-50-many-to-many-mmt">facebook/mbart-large-50-many-to-many-mmt (Recommended)</option>
                        <option value="google/mt5-small">google/mt5-small</option>
                      </optgroup>
                      {Array.isArray(models) && models.length > 0 && (
                        <optgroup label="Trained Checkpoints (Registry)">
                          {models.map(m => m && (
                            <option key={m.id} value={m.checkpoint_path}>
                              {m.model_name} ({m.version}) - Loss: {m.metrics?.final_loss?.toFixed(4)}
                            </option>
                          ))}
                        </optgroup>
                      )}
                    </select>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                    <div>
                      <label style={{ fontSize: '12px', color: '#94a3b8', display: 'block', marginBottom: '6px' }}>Epochs</label>
                      <input type="number" min="1" value={trainConfig.epochs} onChange={e => setTrainConfig({...trainConfig, epochs: parseInt(e.target.value)})} />
                    </div>
                    <div>
                      <label style={{ fontSize: '12px', color: '#94a3b8', display: 'block', marginBottom: '6px' }}>Batch Size</label>
                      <input type="number" min="1" value={trainConfig.batch_size} onChange={e => setTrainConfig({...trainConfig, batch_size: parseInt(e.target.value)})} />
                    </div>
                  </div>
                  {trainConfig.batch_size > 2 && (
                    <div style={{ background: 'rgba(245, 158, 11, 0.05)', border: '1px dashed rgba(245, 158, 11, 0.25)', padding: '8px 12px', borderRadius: '6px', fontSize: '11px', color: '#fcd34d', marginTop: '-4px' }}>
                      ⚠️ <strong>VRAM OOM Guard (6GB Limit):</strong> Batch size of {trainConfig.batch_size} is large. The trainer will automatically enable <strong>Gradient Accumulation</strong> (physical batch = {trainConfig.max_sequence_length >= 128 ? 1 : 2}) to prevent CUDA Out-Of-Memory.
                    </div>
                  )}
                  <div>
                    <label style={{ fontSize: '12px', color: '#94a3b8', display: 'block', marginBottom: '6px' }}>Learning Rate</label>
                    <input type="number" step="0.000001" value={trainConfig.learning_rate} onChange={e => setTrainConfig({...trainConfig, learning_rate: parseFloat(e.target.value)})} />
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <input type="checkbox" checked={trainConfig.fp16} onChange={e => setTrainConfig({...trainConfig, fp16: e.target.checked})} style={{ width: 'auto' }} />
                    <label style={{ fontSize: '13px' }}>Enable FP16 Mixed Precision (RTX 4050 Optimization)</label>
                  </div>
                  <button type="submit" disabled={gpuStatus.is_locked} style={{ marginTop: '8px', background: gpuStatus.is_locked ? '#4b5563' : undefined }}>
                    <Play size={16} /> {gpuStatus.is_locked ? "GPU Locked (Queued)" : "Approve & Start Training"}
                  </button>
                </form>
              </div>

              {/* Experiment runs comparison */}
              <div className="glass-panel">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                  <h4 style={{ fontSize: '16px', fontWeight: 600, margin: 0 }}>Experiment Tracking (MLflow metrics)</h4>
                  <button 
                    className="secondary" 
                    style={{ 
                      padding: '4px 10px', 
                      fontSize: '11px', 
                      background: 'rgba(239, 68, 68, 0.08)', 
                      color: '#fca5a5', 
                      border: '1px solid rgba(239, 68, 68, 0.25)',
                      height: '26px',
                      minWidth: 'auto',
                      cursor: 'pointer',
                      fontWeight: 600
                    }} 
                    onClick={handlePurgeFailed}
                  >
                    🧹 Purge Failed Runs
                  </button>
                </div>
                {!Array.isArray(experiments) || experiments.length === 0 ? (
                  <div>
                    <p style={{ color: '#64748b', fontSize: '14px', marginBottom: '16px' }}>No runs logged yet. Displaying demo run curves.</p>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                      <SVGLineChart data={[1.2, 0.8, 0.6, 0.45, 0.35, 0.28, 0.22, 0.18, 0.15]} title="Training Loss Curve" color="#8b5cf6" />
                      <SVGLineChart data={[1.4, 0.95, 0.72, 0.58, 0.48, 0.42, 0.39, 0.37, 0.36]} title="Validation Loss Curve" color="#06b6d4" />
                    </div>
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    {experiments.map(run => {
                      if (!run) return null;
                      const isActiveRun = activeTrainingJob && run.run_name && run.run_name.includes(activeTrainingJob.id.slice(0, 8));
                      const progress = isActiveRun && activeTrainingJob.config.progress;
                      
                      return (
                        <div key={run.id} className="glass-panel" style={{ padding: '16px', background: 'rgba(255,255,255,0.01)', borderLeft: isActiveRun ? '3px solid var(--color-primary)' : undefined }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <span style={{ fontWeight: 600 }}>{run.run_name}</span>
                            <span className={`badge ${run.status === 'Completed' ? 'success' : run.status === 'Failed' ? 'danger' : 'info'}`}>
                              {isActiveRun ? activeTrainingJob.status : run.status}
                            </span>
                          </div>
                          <p style={{ fontSize: '12px', color: '#64748b', marginTop: '6px' }}>
                            Epochs: {run.hyperparameters?.epochs} | Batch: {run.hyperparameters?.batch_size} | LR: {run.hyperparameters?.learning_rate}
                          </p>

                          {progress && (
                             <div style={{ marginTop: '12px' }}>
                               <TrainingProgressPanel 
                                 progress={progress} 
                                 jobStatus={activeTrainingJob?.status || 'Running'} 
                                 errorLog={activeTrainingJob?.error_log} 
                                 jobId={activeTrainingJob?.id}
                                 onCancel={cancelJob}
                                 onPause={pauseJob}
                                 onResume={resumeJob}
                                 isCompact={true} 
                               />
                             </div>
                           )}

                          {run.metrics?.loss && (
                            <div style={{ marginTop: '16px' }}>
                              <SVGLineChart data={run.metrics.loss} title="Loss Trend" color="#8b5cf6" />
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Model Registry View */}
        {activeTab === 'registry' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
            <div className="glass-panel">
              <h4 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>Registered Model Checkpoints</h4>
              {!Array.isArray(models) || models.length === 0 ? (
                <div>
                  <p style={{ color: '#64748b', fontSize: '14px', marginBottom: '16px' }}>No models registered yet. Rendering mock models.</p>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', textAlign: 'left' }}>
                        <th style={{ padding: '12px', fontSize: '12px', color: '#94a3b8' }}>Model Name</th>
                        <th style={{ padding: '12px', fontSize: '12px', color: '#94a3b8' }}>Version</th>
                        <th style={{ padding: '12px', fontSize: '12px', color: '#94a3b8' }}>Final Loss</th>
                        <th style={{ padding: '12px', fontSize: '12px', color: '#94a3b8' }}>VRAM Size</th>
                        <th style={{ padding: '12px', fontSize: '12px', color: '#94a3b8' }}>Approval</th>
                        <th style={{ padding: '12px', fontSize: '12px', color: '#94a3b8' }}>Deployment</th>
                        <th style={{ padding: '12px', fontSize: '12px', color: '#94a3b8', textAlign: 'right' }}>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.02)' }}>
                        <td style={{ padding: '12px', fontSize: '14px', fontWeight: 550 }}>translation-model-en-kn</td>
                        <td style={{ padding: '12px', fontSize: '13px' }}>v1.0</td>
                        <td style={{ padding: '12px', fontSize: '13px' }}>0.15</td>
                        <td style={{ padding: '12px', fontSize: '13px' }}>836 MB</td>
                        <td style={{ padding: '12px' }}><span className="badge success">Approved</span></td>
                        <td style={{ padding: '12px' }}><span className="badge info">Deployed</span></td>
                        <td style={{ padding: '12px', textAlign: 'right' }}>
                          <button className="secondary" style={{ padding: '6px 12px', fontSize: '12px' }} disabled>Deploy Active</button>
                        </td>
                      </tr>
                      <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.02)' }}>
                        <td style={{ padding: '12px', fontSize: '14px', fontWeight: 550 }}>translation-model-en-ml</td>
                        <td style={{ padding: '12px', fontSize: '13px' }}>v1.1</td>
                        <td style={{ padding: '12px', fontSize: '13px' }}>0.28</td>
                        <td style={{ padding: '12px', fontSize: '13px' }}>836 MB</td>
                        <td style={{ padding: '12px' }}><span className="badge warning">Pending</span></td>
                        <td style={{ padding: '12px' }}><span className="badge muted">Undeployed</span></td>
                        <td style={{ padding: '12px', textAlign: 'right', display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
                          <button style={{ padding: '6px 12px', fontSize: '12px', background: 'var(--color-success)' }} onClick={() => alert("Model Approved!")}>Approve</button>
                          <button className="secondary" style={{ padding: '6px 12px', fontSize: '12px' }} onClick={() => alert("Approved model deployed!")}>Deploy</button>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              ) : (
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', textAlign: 'left' }}>
                      <th style={{ padding: '12px', fontSize: '12px', color: '#94a3b8' }}>Model Name</th>
                      <th style={{ padding: '12px', fontSize: '12px', color: '#94a3b8' }}>Version</th>
                      <th style={{ padding: '12px', fontSize: '12px', color: '#94a3b8' }}>Metrics</th>
                      <th style={{ padding: '12px', fontSize: '12px', color: '#94a3b8' }}>Approval Status</th>
                      <th style={{ padding: '12px', fontSize: '12px', color: '#94a3b8' }}>Deployment Status</th>
                      <th style={{ padding: '12px', fontSize: '12px', color: '#94a3b8', textAlign: 'right' }}>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Array.isArray(models) && models.map(m => m && (
                      <tr key={m.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.02)' }}>
                        <td style={{ padding: '12px', fontSize: '14px', fontWeight: 550 }}>{m.model_name}</td>
                        <td style={{ padding: '12px', fontSize: '13px' }}>{m.version}</td>
                        <td style={{ padding: '12px', fontSize: '13px' }}>Loss: {m.metrics?.final_loss} | Size: {m.metrics?.model_size_mb} MB</td>
                        <td style={{ padding: '12px' }}>
                          <span className={`badge ${m.approval_status === 'Approved' ? 'success' : 'warning'}`}>{m.approval_status}</span>
                        </td>
                        <td style={{ padding: '12px' }}>
                          <span className={`badge ${m.deployment_status === 'Deployed' ? 'info' : 'muted'}`}>{m.deployment_status}</span>
                        </td>
                        <td style={{ padding: '12px', textAlign: 'right', display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
                          {m.approval_status === 'Pending' && (
                            <button style={{ padding: '6px 12px', fontSize: '12px', background: 'var(--color-success)' }} onClick={() => approveModel(m.id)}>Approve</button>
                          )}
                          {m.approval_status === 'Approved' && m.deployment_status === 'Undeployed' && (
                            <button className="secondary" style={{ padding: '6px 12px', fontSize: '12px' }} onClick={() => deployModel(m.id)}>Deploy</button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        )}

        {/* Sandbox View */}
        {activeTab === 'sandbox' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
            <div className="glass-panel" style={{ maxWidth: '800px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
                <h4 style={{ fontSize: '16px', fontWeight: 600 }}>Interactive Translation Sandbox</h4>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span className="badge muted">Stateless</span>
                  {inferenceStatus?.gpu_fallback_active && (
                    <span className="badge warning">CPU Fallback Active</span>
                  )}
                </div>
              </div>

              <form onSubmit={handleTranslate} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', alignItems: 'center', gap: '16px' }}>
                  <select value={sandboxRequest.src} onChange={e => setSandboxRequest({...sandboxRequest, src: e.target.value})}>
                    <option value="en">English (en)</option>
                    <option value="kn">Kannada (kn)</option>
                    <option value="ml">Malayalam (ml)</option>
                  </select>
                  <ArrowRight size={18} color="#64748b" />
                  <select value={sandboxRequest.tgt} onChange={e => setSandboxRequest({...sandboxRequest, tgt: e.target.value})}>
                    <option value="kn">Kannada (kn)</option>
                    <option value="ml">Malayalam (ml)</option>
                    <option value="en">English (en)</option>
                  </select>
                </div>

                <div>
                  <label style={{ fontSize: '12px', color: '#94a3b8', display: 'block', marginBottom: '6px' }}>Input Text</label>
                  <textarea 
                    rows="4" 
                    value={sandboxRequest.text} 
                    onChange={e => setSandboxRequest({...sandboxRequest, text: e.target.value})} 
                    placeholder="Enter sentence to translate..."
                    required
                  />
                </div>

                <button type="submit" disabled={sandboxLoading} style={{ width: 'fit-content' }}>
                  {sandboxLoading ? "Translating..." : "Translate"}
                </button>
              </form>

              {sandboxResult && (
                <div className="glass-panel" style={{ marginTop: '24px', background: 'rgba(255,255,255,0.02)' }}>
                  <h5 style={{ fontSize: '13px', color: '#94a3b8', marginBottom: '8px' }}>Translation Result</h5>
                  {sandboxResult.error ? (
                    <p style={{ color: 'var(--color-error)' }}>{sandboxResult.error}</p>
                  ) : (
                    <div>
                      <p style={{ fontSize: '18px', fontWeight: 600, color: 'var(--color-secondary)' }}>{sandboxResult.translated_text}</p>
                      <div style={{ display: 'flex', gap: '16px', marginTop: '16px', fontSize: '11px', color: '#64748b' }}>
                        <span>Latency: <strong>{sandboxResult.latency_ms} ms</strong></span>
                        <span>Device: <strong>{sandboxResult.device}</strong></span>
                        <span>Model: <strong>{sandboxResult.model_version}</strong></span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
