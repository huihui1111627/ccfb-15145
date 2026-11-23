// Package convert service layer: the conversion service every storage and
// client call passes through. It adds latency accounting, per-kind fault
// injection and a hard failure contract — an unavailable or failed service
// makes dependent reads/writes fail explicitly, so no half-converted data
// can ever be written.
package convert

import (
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"crv/pkg/types"
)

// Service is the network-facing conversion service.
type Service struct {
	engine *Engine

	mu     sync.RWMutex
	faults map[string]Fault // keyed by kind; "*" means global

	metrics *Metrics
	now     func() time.Time
	// sleep is indirected so tests can run instantly under injected slowdown.
	sleep func(d time.Duration)
}

// Fault describes an injected failure mode for one kind (or all kinds).
type Fault struct {
	Down         bool          // service returns ErrServiceDown immediately
	Slow         bool          // service sleeps AddedLatency per call
	AddedLatency time.Duration
	// FailRate in [0,1] makes individual calls fail; 1 = always fail.
	FailRate float64
}

// Metrics holds service-level counters and latency histograms.
type Metrics struct {
	TotalCalls atomic.Int64
	Failed     atomic.Int64

	mu        sync.Mutex
	latencies map[string]*latencyStats // keyed by kind
}

type latencyStats struct {
	count int64
	sum   time.Duration
	min   time.Duration
	max   time.Duration
}

func NewService(engine *Engine) *Service {
	return &Service{
		engine:  engine,
		faults:  map[string]Fault{},
		metrics: &Metrics{latencies: map[string]*latencyStats{}},
		now:     time.Now,
		sleep:   time.Sleep,
	}
}

// ErrServiceDown is returned when the conversion service has been failed
// deliberately (or its injected failure rate trips).
type ErrServiceDown struct{ Kind string }

func (e *ErrServiceDown) Error() string {
	return fmt.Sprintf("conversion service unavailable for kind %s: request rejected, no data written", e.Kind)
}

// ErrSlowConversion is returned when an injected slowdown exceeds the
// caller deadline. Callers treat it exactly like a down service.
type ErrSlowConversion struct{ Kind string }

func (e *ErrSlowConversion) Error() string {
	return fmt.Sprintf("conversion service timed out for kind %s: request rejected, no data written", e.Kind)
}

// ConvertObject converts obj to targetVersion, applying injected faults and
// recording latency. On failure obj is left unchanged (the engine converts
// on an internal scratch copy anyway).
func (s *Service) ConvertObject(obj *types.Object, targetVersion string) error {
	kind := obj.Kind
	start := s.now()
	s.metrics.TotalCalls.Add(1)

	fault := s.activeFault(kind)
	if fault.Down || fault.FailRate >= 1 {
		s.recordFailure(kind, start)
		return &ErrServiceDown{Kind: kind}
	}
	if fault.FailRate > 0 && deterministicTrip(obj, fault.FailRate) {
		s.recordFailure(kind, start)
		return &ErrServiceDown{Kind: kind}
	}
	if fault.Slow && fault.AddedLatency > 0 {
		s.sleep(fault.AddedLatency)
	}

	if err := s.engine.Convert(obj, targetVersion); err != nil {
		s.recordFailure(kind, start)
		return err
	}
	s.recordLatency(kind, s.now().Sub(start))
	return nil
}

// activeFault merges the global ("*") fault with the kind-specific one.
func (s *Service) activeFault(kind string) Fault {
	s.mu.RLock()
	defer s.mu.RUnlock()
	f := s.faults["*"]
	if kf, ok := s.faults[kind]; ok {
		if kf.Down {
			f.Down = true
		}
		if kf.Slow {
			f.Slow = true
			f.AddedLatency = kf.AddedLatency
		}
		if kf.FailRate > f.FailRate {
			f.FailRate = kf.FailRate
		}
	}
	return f
}

// deterministicTrip makes fail-rate injection reproducible without global
// randomness: the object's resource version selects the trip window.
func deterministicTrip(obj *types.Object, rate float64) bool {
	if rate <= 0 {
		return false
	}
	window := int64(1000 * rate)
	if window <= 0 {
		return false
	}
	return obj.Metadata.ResourceVersion%1000 < window
}

func (s *Service) recordFailure(kind string, start time.Time) {
	s.metrics.Failed.Add(1)
	s.recordLatency(kind, s.now().Sub(start))
}

func (s *Service) recordLatency(kind string, d time.Duration) {
	s.metrics.mu.Lock()
	defer s.metrics.mu.Unlock()
	st, ok := s.metrics.latencies[kind]
	if !ok {
		st = &latencyStats{min: d, max: d}
		s.metrics.latencies[kind] = st
	}
	st.count++
	st.sum += d
	if d < st.min {
		st.min = d
	}
	if d > st.max {
		st.max = d
	}
}

// ----------------------------------------------------------- fault control --

// SetFault installs (or clears, with the zero value) a fault for a kind.
// Pass kind "*" to fault every kind at once.
func (s *Service) SetFault(kind string, f Fault) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if f == (Fault{}) {
		delete(s.faults, kind)
		return
	}
	s.faults[kind] = f
}

// ClearFaults removes every injected fault.
func (s *Service) ClearFaults() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.faults = map[string]Fault{}
}

// Faults returns a snapshot of the active faults.
func (s *Service) Faults() map[string]Fault {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make(map[string]Fault, len(s.faults))
	for k, v := range s.faults {
		out[k] = v
	}
	return out
}

// LatencyReport is one row of the service latency statistics.
type LatencyReport struct {
	Kind    string        `json:"kind"`
	Calls   int64         `json:"calls"`
	Average time.Duration `json:"average"`
	Min     time.Duration `json:"min"`
	Max     time.Duration `json:"max"`
}

// Report returns latency stats sorted by kind, plus total counters.
func (s *Service) Report() (rows []LatencyReport, total, failed int64) {
	s.metrics.mu.Lock()
	defer s.metrics.mu.Unlock()
	rows = make([]LatencyReport, 0, len(s.metrics.latencies))
	for kind, st := range s.metrics.latencies {
		avg := time.Duration(0)
		if st.count > 0 {
			avg = st.sum / time.Duration(st.count)
		}
		rows = append(rows, LatencyReport{
			Kind: kind, Calls: st.count, Average: avg, Min: st.min, Max: st.max,
		})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Kind < rows[j].Kind })
	return rows, s.metrics.TotalCalls.Load(), s.metrics.Failed.Load()
}
