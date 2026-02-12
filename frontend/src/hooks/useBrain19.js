import { useState, useEffect, useRef, useCallback } from 'react';

const WS_URL = `ws://${window.location.hostname}:8019/ws`;
const API_BASE = 'http://172.16.16.104:8019/api';

export function useBrain19() {
  const [snapshot, setSnapshot] = useState(null);
  const [connected, setConnected] = useState(false);
  const [loading, setLoading] = useState(true);
  const [lastAnswer, setLastAnswer] = useState(null);
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setLoading(false);
      console.log('[Brain19] WebSocket connected');
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'snapshot') {
          setSnapshot(msg.data);
        } else if (msg.type === 'answer') {
          setLastAnswer(msg.data);
        } else if (msg.type === 'ingested') {
          ws.send(JSON.stringify({ command: 'snapshot' }));
        }
      } catch (e) {
        console.error('[Brain19] Parse error:', e);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      console.log('[Brain19] WebSocket disconnected, reconnecting...');
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => ws.close();
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const ask = useCallback((question) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      setLastAnswer(null);
      wsRef.current.send(JSON.stringify({ command: 'ask', text: question }));
    }
  }, []);

  const ingest = useCallback((text) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command: 'ingest', text }));
    }
  }, []);

  const refreshSnapshot = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command: 'snapshot' }));
    }
  }, []);

  const fetchSnapshot = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/snapshot`);
      const data = await res.json();
      setSnapshot(data);
      setLoading(false);
    } catch (e) {
      console.error('[Brain19] REST fetch failed:', e);
    }
  }, []);

  return { snapshot, connected, loading, lastAnswer, ask, ingest, refreshSnapshot, fetchSnapshot };
}
