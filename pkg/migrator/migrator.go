// Package migrator performs background storage-version migration.
//
// Switching a kind's storage version takes effect for new writes
// immediately; existing objects are then rewritten one by one. Each rewrite
// uses an optimistic CAS (storage.MigrateOne) so concurrent client traffic
// never loses updates: if a client writes an object first, the migrator
// observes the new resource version and skips it. Reads and writes continue
// throughout — an object is always served in whatever version it currently
// has, converted on demand.
//
// A version may only be removed from a kind's definition after zero stored
// objects remain in it.
package migrator

import (
	"fmt"
	"sync"
	"time"

	"crv/pkg/schema"
	"crv/pkg/storage"
)

// State of a migration job.
type State string

const (
	StateRunning   State = "running"
	StateCompleted State = "completed"
	StateFailed    State = "failed" // blocked by conversion-service fault
)

// Progress is the externally visible migration state for one kind.
type Progress struct {
	Kind            string    `json:"kind"`
	FromVersion     string    `json:"from_version"`
	ToVersion       string    `json:"to_version"`
	Total           int       `json:"total"`
	Migrated        int       `json:"migrated"`
	Skipped         int       `json:"skipped"`
	FailedAttempts  int       `json:"failed_attempts"`
	State           State     `json:"state"`
	StartedAt       time.Time `json:"started_at"`
	FinishedAt      time.Time `json:"finished_at,omitempty"`
	LastError       string    `json:"last_error,omitempty"`
}

func (p Progress) Percent() float64 {
	if p.Total == 0 {
		return 100
	}
	return float64(p.Migrated+p.Skipped) / float64(p.Total) * 100
}

// Migrator runs background storage-version migrations.
type Migrator struct {
	reg *schema.Registry
	st  *storage.Store

	mu       sync.Mutex
	jobs     map[string]*job
	stopCh   chan struct{}
	stepInterval time.Duration // pacing between objects (0 = as fast as possible)
}

type job struct {
	progress Progress
	stopCh   chan struct{}
}

func New(reg *schema.Registry, st *storage.Store) *Migrator {
	return &Migrator{
		reg:          reg,
		st:           st,
		jobs:         map[string]*job{},
		stopCh:       make(chan struct{}),
	}
}

// WithStepInterval sets a delay between migrated objects to make progress
// observable in the UI during acceptance runs.
func (m *Migrator) WithStepInterval(d time.Duration) *Migrator {
	m.stepInterval = d
	return m
}

// SwitchStorage changes the storage version of a kind and starts a
// background migration of all objects still stored in the old version.
// New writes immediately use newVersion; the migration runs concurrently
// with normal traffic.
func (m *Migrator) SwitchStorage(kind, newVersion string) (*Progress, error) {
	k, err := m.reg.Get(kind)
	if err != nil {
		return nil, err
	}
	oldVersion := k.StorageVersion
	if oldVersion == newVersion {
		return nil, fmt.Errorf("kind %s already stores in %s", kind, newVersion)
	}
	if _, err := k.VersionIndex(newVersion); err != nil {
		return nil, err
	}
	if err := m.reg.SetStorageVersion(kind, newVersion); err != nil {
		return nil, err
	}

	j := &job{
		progress: Progress{
			Kind: kind, FromVersion: oldVersion, ToVersion: newVersion,
			Total:     m.st.Count(kind),
			State:     StateRunning,
			StartedAt: time.Now(),
		},
		stopCh: make(chan struct{}),
	}
	m.mu.Lock()
	if existing, ok := m.jobs[kind]; ok && existing.progress.State == StateRunning {
		m.mu.Unlock()
		return nil, fmt.Errorf("migration for %s already running", kind)
	}
	m.jobs[kind] = j
	m.mu.Unlock()

	go m.run(j)
	return m.Progress(kind), nil
}

func (m *Migrator) run(j *job) {
	for {
		select {
		case <-m.stopCh:
			return
		case <-j.stopCh:
			return
		default:
		}
		keys := m.st.KeysForVersion(j.progress.Kind, j.progress.FromVersion)
		if len(keys) == 0 {
			m.finish(j, StateCompleted, "")
			return
		}
		key := keys[0]
		ns, name := splitKey(key)
		migrated, err := m.st.MigrateOne(j.progress.Kind, ns, name, j.progress.FromVersion)
		m.mu.Lock()
		switch {
		case err != nil:
			j.progress.FailedAttempts++
			j.progress.LastError = err.Error()
		case migrated:
			j.progress.Migrated++
		default:
			j.progress.Skipped++
		}
		m.mu.Unlock()

		if err != nil {
			// Conversion service down/slow: back off and retry rather
			// than writing anything. Traffic-facing reads/writes fail
			// explicitly meanwhile; no bad data is produced.
			if !wait(250*time.Millisecond, j.stopCh, m.stopCh) {
				return
			}
			continue
		}
		if m.stepInterval > 0 {
			if !wait(m.stepInterval, j.stopCh, m.stopCh) {
				return
			}
		}
	}
}

func (m *Migrator) finish(j *job, state State, lastErr string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	j.progress.State = state
	j.progress.LastError = lastErr
	j.progress.FinishedAt = time.Now()
}

// Progress returns a snapshot of a kind's migration progress.
func (m *Migrator) Progress(kind string) *Progress {
	m.mu.Lock()
	defer m.mu.Unlock()
	j, ok := m.jobs[kind]
	if !ok {
		return nil
	}
	p := j.progress
	return &p
}

// AllProgress returns progress snapshots for every migration, sorted by kind.
func (m *Migrator) AllProgress() []Progress {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]Progress, 0, len(m.jobs))
	for _, j := range m.jobs {
		out = append(out, j.progress)
	}
	// stable order without importing sort twice
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j-1].Kind > out[j].Kind; j-- {
			out[j-1], out[j] = out[j], out[j-1]
		}
	}
	return out
}

// CanRemoveVersion is the safety gate for dropping a version definition:
// no stored object may still be shaped like it, and no migration may be
// running that touches it.
func (m *Migrator) CanRemoveVersion(kind, version string) error {
	if count := m.st.CountByStorageVersion(kind)[version]; count > 0 {
		return fmt.Errorf("cannot remove version %s of %s: %d stored objects still use it", version, kind, count)
	}
	if p := m.Progress(kind); p != nil && p.State == StateRunning &&
		(p.FromVersion == version || p.ToVersion == version) {
		return fmt.Errorf("cannot remove version %s of %s: migration in progress", version, kind)
	}
	return nil
}

// RemoveVersion removes a version only after the drain gate passes.
func (m *Migrator) RemoveVersion(kind, version string) error {
	if err := m.CanRemoveVersion(kind, version); err != nil {
		return err
	}
	return m.reg.RemoveVersion(kind, version)
}

// Stop halts all background work (used by test teardown).
func (m *Migrator) Stop() {
	close(m.stopCh)
}

func splitKey(key string) (string, string) {
	for i := 0; i < len(key); i++ {
		if key[i] == '/' {
			return key[:i], key[i+1:]
		}
	}
	return "", key
}

// wait sleeps for d unless the job or migrator is stopped; returns false
// when stopped before the timer fired.
func wait(d time.Duration, jobStop, globalStop chan struct{}) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-jobStop:
		return false
	case <-globalStop:
		return false
	}
}
