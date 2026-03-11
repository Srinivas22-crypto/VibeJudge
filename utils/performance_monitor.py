import time
import logging
import tracemalloc
from contextlib import contextmanager
from typing import Dict, List
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Tracks timing and memory for each pipeline stage.

    Usage:
        monitor = PerformanceMonitor(podcast_id="abc123")

        with monitor.track("Transcription"):
            result = transcriber.transcribe(audio_path)

        with monitor.track("Sentiment"):
            sent = analyzer.analyze_text(text)

        monitor.print_summary()
        monitor.save_report("data/results/perf_abc123.json")
    """

    def __init__(self, podcast_id: str = "unknown", audio_duration: float = 0):
        self.podcast_id     = podcast_id
        self.audio_duration = audio_duration  # seconds
        self.stages: List[Dict] = []
        self._start_total   = time.perf_counter()

    @contextmanager
    def track(self, stage_name: str):
        """Context manager to time and measure memory for a stage"""
        tracemalloc.start()
        t_start = time.perf_counter()

        try:
            yield
        finally:
            elapsed = time.perf_counter() - t_start
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            self.stages.append({
                "stage":       stage_name,
                "duration_s":  round(elapsed, 3),
                "peak_mem_mb": round(peak_mem / 1024 / 1024, 2)
            })
            logger.info(
                f"[Perf] {stage_name:25s} | "
                f"{elapsed:.2f}s | "
                f"{peak_mem/1024/1024:.1f} MB"
            )

    def total_duration(self) -> float:
        return round(time.perf_counter() - self._start_total, 3)

    def realtime_factor(self) -> float:
        if self.audio_duration <= 0:
            return 0.0
        return round(self.total_duration() / self.audio_duration, 3)

    def summary(self) -> Dict:
        total   = self.total_duration()
        rt      = self.realtime_factor()
        passed  = total / max(self.audio_duration, 1) < 2.0

        return {
            "podcast_id":         self.podcast_id,
            "audio_duration_s":   self.audio_duration,
            "total_duration_s":   total,
            "realtime_factor":    rt,
            "target_met_2x":      passed,
            "stages":             self.stages,
            "peak_memory_mb":     max((s["peak_mem_mb"] for s in self.stages), default=0),
            "timestamp":          datetime.now().isoformat()
        }

    def print_summary(self) -> None:
        summ = self.summary()
        print("\n" + "="*55)
        print(f"  PERFORMANCE REPORT — {self.podcast_id}")
        print("="*55)
        print(f"  {'Stage':<25} {'Time (s)':>10} {'Peak MB':>10}")
        print("  " + "-"*45)
        for s in self.stages:
            print(f"  {s['stage']:<25} {s['duration_s']:>10.2f} {s['peak_mem_mb']:>10.1f}")
        print("  " + "-"*45)
        print(f"  {'TOTAL':<25} {summ['total_duration_s']:>10.2f}")
        print(f"\n  Audio Duration : {self.audio_duration:.0f}s "
              f"({self.audio_duration/60:.1f} min)")
        print(f"  Real-Time Factor: {summ['realtime_factor']:.2f}×")
        target_str = "✓ PASS" if summ["target_met_2x"] else "✗ FAIL"
        print(f"  Target (<2× RT): {target_str}")
        print("="*55)

    def save_report(self, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.summary(), f, indent=2)
        logger.info(f"✓ Performance report saved to {output_path}")
