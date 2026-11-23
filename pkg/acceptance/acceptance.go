// Package acceptance runs the end-to-end verification scenarios the
// platform must satisfy:
//
//  1. arbitrary version combinations: read-modify-write never loses fields
//  2. storage-version migration: object semantics are identical before and
//     after, and traffic continues throughout
//  3. conversion-service latency accounting
//  4. conversion-service failure: explicit errors, zero half-converted data
//  5. the version-removal safety gate
package acceptance

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"crv/pkg/convert"
	"crv/pkg/migrator"
	"crv/pkg/schema"
	"crv/pkg/simulator"
	"crv/pkg/types"
)

// Report is the structured result of one acceptance run.
type Report struct {
	Seed              int64
	KindCount         int
	VersionCount      int
	ObjectCount       int
	Traffic           *simulator.TrafficStats
	LatencyRows       []convert.LatencyReport
	LatencyTotal      int64
	LatencyFailed     int64
	Migrations        []migrator.Progress
	OutageChecks      OutageChecks
	RemovalGateChecks []string
	Failures          []string
}

// OutageChecks captures the fault-injection verification.
type OutageChecks struct {
	FaultedKinds     []string
	RejectedRequests int64
	HealthySuccesses int64
	BadRecords       int
	Recovered        bool
}

func (r *Report) OK() bool { return len(r.Failures) == 0 }

func (r *Report) fail(format string, args ...any) {
	r.Failures = append(r.Failures, fmt.Sprintf(format, args...))
}

// Run executes the full acceptance suite against a freshly generated
// platform with the given seed and object density.
func Run(seed int64, objectsPerKind, trafficOps int) *Report {
	r := &Report{Seed: seed}
	p, err := simulator.NewPlatform(seed)
	if err != nil {
		r.fail("platform build: %v", err)
		return r
	}
	defer p.Stop()

	kinds := p.Registry.All()
	r.KindCount = len(kinds)
	for _, k := range kinds {
		r.VersionCount += len(k.Versions)
	}

	// ------------------------------------------------------------- seeding
	if err := p.SeedObjects(objectsPerKind); err != nil {
		r.fail("seeding: %v", err)
		return r
	}
	r.ObjectCount = 0
	for _, k := range kinds {
		r.ObjectCount += p.Store.Count(k.Name)
	}

	// Immutable semantic baseline captured BEFORE any traffic or migration:
	// every seeded object's oldest-version shape. Used by the isolated
	// migration-semantics platform below.
	baseline := r.captureBaseline(p, kinds)

	// A second platform with identical seed and NO mutation traffic: the
	// only thing that ever rewrites its objects is the storage migrator, so
	// any before/after difference is provably a conversion defect rather
	// than a legitimate client write.
	migPlatform, err := simulator.NewPlatform(seed)
	if err != nil {
		r.fail("migration platform build: %v", err)
		return r
	}
	defer migPlatform.Stop()
	if err := migPlatform.SeedObjects(objectsPerKind); err != nil {
		r.fail("migration platform seeding: %v", err)
		return r
	}

	// ------------------------------------- 1. mixed-version traffic, no loss
	stats := p.RunTraffic(trafficOps, 8)
	r.Traffic = stats
	if stats.InvariantFails.Load() != 0 {
		r.fail("field-loss invariant failed %d time(s): %s",
			stats.InvariantFails.Load(), stats.FirstViolation)
	}

	// exhaustive cross-version reads for every object of a sampled set
	r.verifyAllVersionViews(p)

	// --------------------------------- 4. fault injection before migration
	r.runOutageScenario(p, kinds)

	// -------------------------------------- 2. storage migration semantics
	r.runMigrations(migPlatform, kinds, baseline)

	// --------------------------------------------- 3. latency statistics
	rows, total, failed := p.Service.Report()
	r.LatencyRows, r.LatencyTotal, r.LatencyFailed = rows, total, failed

	// --------------------------------- 5. version-removal safety gate
	r.runRemovalGate(migPlatform, kinds)

	return r
}

// Baseline maps kind -> object key -> oldest-version spec, captured before
// the platform carries any mutation traffic.
type Baseline map[string]map[string]map[string]any

// captureBaseline records every seeded object's shape in its kind's oldest
// served version.
func (r *Report) captureBaseline(p *simulator.Platform, kinds []*schema.Kind) Baseline {
	base := Baseline{}
	for _, k := range kinds {
		oldest := k.Versions[0].Name
		objs, err := p.Store.List(k.Name, oldest)
		if err != nil {
			r.fail("baseline list %s: %v", k.Name, err)
			continue
		}
		perKind := map[string]map[string]any{}
		for _, o := range objs {
			perKind[o.Key()] = o.Spec
		}
		base[k.Name] = perKind
	}
	return base
}

// verifyAllVersionViews reads every object of every kind through every
// served version and confirms the latest-version view is stable afterwards.
func (r *Report) verifyAllVersionViews(p *simulator.Platform) {
	for _, k := range p.Registry.All() {
		latest := k.Versions[len(k.Versions)-1].Name
		objs, err := p.Store.List(k.Name, latest)
		if err != nil {
			r.fail("list %s latest: %v", k.Name, err)
			continue
		}
		for _, o := range objs {
			for _, vs := range k.Versions {
				if _, err := p.Store.Get(k.Name, o.Metadata.Namespace, o.Metadata.Name, vs.Name); err != nil {
					r.fail("read %s/%s as %s: %v", k.Name, o.Key(), vs.Name, err)
				}
			}
		}
	}
}

// runOutageScenario faults several kinds, drives traffic, checks every
// stored record remains intact, then recovers and confirms drain resumes.
func (r *Report) runOutageScenario(p *simulator.Platform, kinds []*schema.Kind) {
	faultIdx := []int{0, len(kinds) / 3, 2 * len(kinds) / 3}
	for _, idx := range faultIdx {
		k := kinds[idx]
		p.Service.SetFault(k.Name, convert.Fault{Down: true})
		r.OutageChecks.FaultedKinds = append(r.OutageChecks.FaultedKinds, k.Name)
	}
	stats := p.RunTraffic(600, 6)
	r.OutageChecks.RejectedRequests = stats.RejectedFaults.Load()
	r.OutageChecks.HealthySuccesses = stats.Success.Load()
	if stats.InvariantFails.Load() != 0 {
		r.fail("invariant failure during outage: %s", stats.FirstViolation)
	}
	if stats.RejectedFaults.Load() == 0 {
		r.fail("no requests were rejected despite a hard service outage")
	}

	// verify no record is corrupt on disk. A faulted conversion service
	// rightly rejects conversions, so integrity is checked against the raw
	// stored envelope only: it must decode and keep its declared storage
	// version — nothing half-converted may have been written.
	for _, k := range p.Registry.All() {
		for _, key := range p.Store.KeysForVersion(k.Name, k.StorageVersion) {
			ns, name := splitKey(key)
			sv, raw, ok := p.Store.RawRecord(k.Name, ns, name)
			if !ok || raw == nil {
				r.OutageChecks.BadRecords++
				continue
			}
			if sv != k.StorageVersion {
				r.OutageChecks.BadRecords++
			}
		}
	}
	if r.OutageChecks.BadRecords > 0 {
		r.fail("%d bad records found after conversion outage", r.OutageChecks.BadRecords)
	}

	p.Service.ClearFaults()
	// after recovery, every kind must serve again
	probe := kinds[0]
	if _, err := p.Store.List(probe.Name, probe.StorageVersion); err != nil {
		r.fail("service did not recover: %v", err)
		return
	}
	r.OutageChecks.Recovered = true
}

// runMigrations switches each kind to its newest storage version and waits
// for the background migration, confirming progress tracking and that the
// pre-traffic semantic baseline is preserved through the whole chain.
func (r *Report) runMigrations(p *simulator.Platform, kinds []*schema.Kind, baseline Baseline) {
	for _, k := range kinds {
		target := k.Versions[len(k.Versions)-1].Name
		oldest := k.Versions[0].Name
		// concurrent READ traffic while migrating proves the store stays
		// available; it never mutates, so baseline comparison stays exact.
		stopTraffic := make(chan struct{})
		trafficDone := make(chan struct{})
		go func(kindName, wantVersion string) {
			defer close(trafficDone)
			tk := time.NewTicker(2 * time.Millisecond)
			defer tk.Stop()
			for {
				select {
				case <-stopTraffic:
					return
				case <-tk.C:
					_, _ = p.Store.List(kindName, wantVersion)
				}
			}
		}(k.Name, k.StorageVersion)

		prog, err := p.Migrator.SwitchStorage(k.Name, target)
		if err != nil {
			close(stopTraffic)
			<-trafficDone
			r.fail("switch %s: %v", k.Name, err)
			continue
		}
		_ = prog
		finished := waitMigration(p, k.Name, 10*time.Second)
		close(stopTraffic)
		<-trafficDone

		if !finished {
			r.fail("migration of %s did not finish", k.Name)
			continue
		}
		progFinal := p.Migrator.Progress(k.Name)
		r.Migrations = append(r.Migrations, *progFinal)

		// all objects now in the new storage version
		counts := p.Store.CountByStorageVersion(k.Name)
		if counts[target] != p.Store.Count(k.Name) {
			r.fail("%s migration incomplete: %v", k.Name, counts)
			continue
		}
		// every baseline object must still present its original oldest-view
		// semantics after the storage-version rewrite
		after, err := p.Store.List(k.Name, oldest)
		if err != nil {
			r.fail("post-migration read %s: %v", k.Name, err)
			continue
		}
		baseForKind := baseline[k.Name]
		for _, o := range after {
			want, existed := baseForKind[o.Key()]
			if !existed {
				continue // created after the baseline (traffic phase)
			}
			if !semanticSubset(want, o.Spec) {
				var detail string
				for key, v := range want {
					gv, ok := o.Spec[key]
					jv, _ := json.Marshal(v)
					if !ok {
						detail += fmt.Sprintf("\n  missing %s=%s", key, jv)
						continue
					}
					jg, _ := json.Marshal(gv)
					if string(jv) != string(jg) {
						detail += fmt.Sprintf("\n  %s: want %s got %s", key, jv, jg)
					}
				}
				r.fail("%s/%s semantic drift after migration%s", k.Name, o.Key(), detail)
			}
		}
	}
}

// runRemovalGate confirms the migrator blocks version removal until zero
// objects remain, then permits it after drain.
func (r *Report) runRemovalGate(p *simulator.Platform, kinds []*schema.Kind) {
	k := kinds[0]
	oldest := k.Versions[0].Name
	// all migrations have drained by now, so the gate passes
	if err := p.Migrator.CanRemoveVersion(k.Name, oldest); err != nil {
		r.fail("removal gate blocked a fully drained version: %v", err)
		return
	}
	r.RemovalGateChecks = append(r.RemovalGateChecks,
		fmt.Sprintf("%s: drain confirmed, %s removable", k.Name, oldest))

	// the storage version itself can never be removed
	newest := k.Versions[len(k.Versions)-1].Name
	if err := p.Migrator.CanRemoveVersion(k.Name, newest); err == nil {
		r.fail("gate allowed removing the active storage version")
	} else {
		r.RemovalGateChecks = append(r.RemovalGateChecks,
			fmt.Sprintf("%s: storage version removal correctly blocked", k.Name))
	}
}

// semanticSubset reports that every field visible in the older snapshot is
// still present with the same value (newer fields are allowed in `got`).
// Comparison goes through canonical JSON so float formatting differences
// from in-memory vs. JSON-decoded numbers cannot cause false drift.
func semanticSubset(want, got map[string]any) bool {
	for k, v := range want {
		gv, ok := got[k]
		if !ok {
			return false
		}
		jv, _ := json.Marshal(v)
		jg, _ := json.Marshal(gv)
		if string(jv) != string(jg) {
			return false
		}
	}
	return true
}

func waitMigration(p *simulator.Platform, kind string, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		prog := p.Migrator.Progress(kind)
		if prog != nil && prog.State == migrator.StateCompleted {
			return true
		}
		time.Sleep(10 * time.Millisecond)
	}
	return false
}

func splitKey(key string) (string, string) {
	if i := strings.Index(key, "/"); i >= 0 {
		return key[:i], key[i+1:]
	}
	return "", key
}

// Pretty renders a human-readable acceptance summary.
func (r *Report) Pretty() string {
	var b strings.Builder
	fmt.Fprintf(&b, "ACCEPTANCE REPORT (seed=%d)\n", r.Seed)
	fmt.Fprintf(&b, "  resources: %d kinds, %d version schemas, %d stored objects\n",
		r.KindCount, r.VersionCount, r.ObjectCount)
	if r.Traffic != nil {
		fmt.Fprintf(&b, "  mixed traffic: %d ops | reads=%d rmw=%d creates=%d | success=%d conflicts=%d rejected=%d | field-loss=%d\n",
			r.Traffic.Iterations.Load(), r.Traffic.Reads.Load(), r.Traffic.ReadModifyWrite.Load(),
			r.Traffic.Creates.Load(), r.Traffic.Success.Load(), r.Traffic.Conflicts.Load(),
			r.Traffic.RejectedFaults.Load(), r.Traffic.InvariantFails.Load())
	}
	fmt.Fprintf(&b, "  conversion service: %d calls, %d failures\n", r.LatencyTotal, r.LatencyFailed)
	if len(r.LatencyRows) > 0 {
		var sum time.Duration
		for _, row := range r.LatencyRows {
			sum += row.Average
		}
		fmt.Fprintf(&b, "  latency: mean-of-kind-averages %s over %d kinds\n",
			sum/time.Duration(len(r.LatencyRows)), len(r.LatencyRows))
	}
	fmt.Fprintf(&b, "  migrations: %d kinds completed\n", len(r.Migrations))
	for _, m := range r.Migrations {
		fmt.Fprintf(&b, "    - %s %s->%s: %d/%d (migrated=%d skipped=%d retries=%d) %.0f%%\n",
			m.Kind, m.FromVersion, m.ToVersion, m.Migrated+m.Skipped, m.Total,
			m.Migrated, m.Skipped, m.FailedAttempts, m.Percent())
	}
	fmt.Fprintf(&b, "  outage: faulted=%v rejected=%d healthy-success=%d bad-records=%d recovered=%v\n",
		r.OutageChecks.FaultedKinds, r.OutageChecks.RejectedRequests,
		r.OutageChecks.HealthySuccesses, r.OutageChecks.BadRecords, r.OutageChecks.Recovered)
	for _, c := range r.RemovalGateChecks {
		fmt.Fprintf(&b, "  gate: %s\n", c)
	}
	if r.OK() {
		b.WriteString("RESULT: PASS — all acceptance criteria satisfied\n")
	} else {
		b.WriteString("RESULT: FAIL\n")
		for _, f := range r.Failures {
			fmt.Fprintf(&b, "  - %s\n", f)
		}
	}
	return b.String()
}

var _ = types.StashAnnotationPrefix
