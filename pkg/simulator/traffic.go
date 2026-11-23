package simulator

import (
	"errors"
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"crv/pkg/client"
	"crv/pkg/convert"
	"crv/pkg/migrator"
	"crv/pkg/schema"
	"crv/pkg/storage"
	"crv/pkg/types"
)

// Platform wires together every component of the simulated cluster.
type Platform struct {
	Registry *schema.Registry
	Engine   *convert.Engine
	Service  *convert.Service
	Store    *storage.Store
	Migrator *migrator.Migrator
	Rand     *rand.Rand
	randMu   sync.Mutex
}

// NewPlatform builds the full component stack from generated definitions.
func NewPlatform(seed int64) (*Platform, error) {
	reg, err := GenerateRegistry(seed)
	if err != nil {
		return nil, err
	}
	engine := convert.NewEngine(reg)
	svc := convert.NewService(engine)
	st := storage.New(reg, svc)
	mg := migrator.New(reg, st)
	return &Platform{
		Registry: reg, Engine: engine, Service: svc,
		Store: st, Migrator: mg, Rand: rand.New(rand.NewSource(seed + 1)),
	}, nil
}

// Stop releases background goroutines.
func (p *Platform) Stop() { p.Migrator.Stop() }

func (p *Platform) random() *rand.Rand {
	p.randMu.Lock()
	defer p.randMu.Unlock()
	// rng is used sequentially under the lock; returning it is safe.
	return p.Rand
}

// SeedObjects creates objectsPerKind objects per kind, each inserted by a
// client of a random served version so stored data is immediately
// heterogeneous in origin (though normalized to the storage version).
func (p *Platform) SeedObjects(objectsPerKind int) error {
	for _, k := range p.Registry.All() {
		rng := rand.New(rand.NewSource(p.Rand.Int63()))
		for i := 0; i < objectsPerKind; i++ {
			v := k.Versions[rng.Intn(len(k.Versions))].Name
			obj, err := NewObject(k, v, "default",
				fmt.Sprintf("%s-%04d", lower(k.Name), i), rng)
			if err != nil {
				return err
			}
			c := client.New(p.Store, k.Name, v, client.Preserving)
			if _, err := c.Create(obj); err != nil {
				return fmt.Errorf("seed %s: %w", k.Name, err)
			}
		}
	}
	return nil
}

// TrafficStats summarizes one mixed-traffic run. Every counter is atomic
// because worker goroutines increment them concurrently.
type TrafficStats struct {
	Iterations      atomic.Int64
	Reads           atomic.Int64
	ReadModifyWrite atomic.Int64
	Creates         atomic.Int64
	Success         atomic.Int64
	RejectedFaults  atomic.Int64 // expected failures while service is faulted
	Conflicts       atomic.Int64
	InvariantFails  atomic.Int64
	FirstViolation  string
	violMu          sync.Mutex
}

// createCounter guarantees unique generated object names across workers.
var createCounter atomic.Int64

func (s *TrafficStats) recordViolation(msg string) {
	s.InvariantFails.Add(1)
	s.violMu.Lock()
	if s.FirstViolation == "" {
		s.FirstViolation = msg
	}
	s.violMu.Unlock()
}

// RunTraffic drives mixed-version client traffic across all kinds for
// iterations total operations, spread over workers goroutines.
//
// Per-operation invariant: fields the chosen client version cannot
// represent (i.e. newer-version fields hidden in stash annotations) must be
// identical before and after every read-modify-write — old clients never
// clobber new fields.
func (p *Platform) RunTraffic(iterations, workers int) *TrafficStats {
	stats := &TrafficStats{}
	kinds := p.Registry.All()
	var wg sync.WaitGroup
	var counter int64

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(id)*7919 + 42))
			for {
				n := atomic.AddInt64(&counter, 1)
				if n > int64(iterations) {
					return
				}
				stats.Iterations.Add(1)
				k := kinds[rng.Intn(len(kinds))]
				switch rng.Intn(10) {
				case 0, 1, 2: // 30% cross-version reads
					stats.Reads.Add(1)
					p.crossVersionRead(k, rng, stats)
				case 9: // 10% creates
					stats.Creates.Add(1)
					p.createFromRandomVersion(k, rng, stats)
				default: // 60% RMW
					stats.ReadModifyWrite.Add(1)
					p.readModifyWrite(k, rng, stats)
				}
			}
		}(w)
	}
	wg.Wait()
	return stats
}

func (p *Platform) crossVersionRead(k *schema.Kind, rng *rand.Rand, stats *TrafficStats) {
	objs, err := p.Store.List(k.Name, k.Versions[rng.Intn(len(k.Versions))].Name)
	if err != nil {
		if isServiceError(err) {
			stats.RejectedFaults.Add(1)
			return
		}
		stats.recordViolation(fmt.Sprintf("list %s unexpected error: %v", k.Name, err))
		return
	}
	if len(objs) == 0 {
		return
	}
	// every served version must render the same object without panicking
	first := objs[rng.Intn(len(objs))]
	for _, vs := range k.Versions {
		if _, err := p.Store.Get(k.Name, first.Metadata.Namespace, first.Metadata.Name, vs.Name); err != nil {
			if isServiceError(err) {
				stats.RejectedFaults.Add(1)
				return
			}
			stats.recordViolation(fmt.Sprintf("read %s in %s: %v", k.Name, vs.Name, err))
		}
	}
	stats.Success.Add(1)
}

func (p *Platform) createFromRandomVersion(k *schema.Kind, rng *rand.Rand, stats *TrafficStats) {
	v := k.Versions[rng.Intn(len(k.Versions))].Name
	// globally unique name: timestamp modulo is collision-prone under load
	name := fmt.Sprintf("%s-traffic-%d-%d", lower(k.Name), time.Now().UnixNano(), createCounter.Add(1))
	obj, err := NewObject(k, v, "default", name, rng)
	if err != nil {
		stats.recordViolation(err.Error())
		return
	}
	c := client.New(p.Store, k.Name, v, client.Preserving)
	if _, err := c.Create(obj); err != nil {
		if isServiceError(err) {
			stats.RejectedFaults.Add(1)
			return
		}
		stats.recordViolation(fmt.Sprintf("create %s: %v", k.Name, err))
		return
	}
	stats.Success.Add(1)
}

func (p *Platform) readModifyWrite(k *schema.Kind, rng *rand.Rand, stats *TrafficStats) {
	version := k.Versions[rng.Intn(len(k.Versions))].Name
	c := client.New(p.Store, k.Name, version, client.Preserving)
	existing, err := c.List()
	if err != nil {
		if isServiceError(err) {
			stats.RejectedFaults.Add(1)
			return
		}
		stats.recordViolation(fmt.Sprintf("rmw list %s: %v", k.Name, err))
		return
	}
	if len(existing) == 0 {
		return
	}
	obj := existing[rng.Intn(len(existing))]

	// Snapshot everything this client version does NOT know. Those fields
	// live in stash annotations and must survive the write untouched.
	unknownBefore := snapshotUnknown(k, version, obj)

	// mutate one field the client does understand
	vs, _ := k.Version(version)
	specFields := specOnly(vs.Fields)
	if len(specFields) == 0 {
		return
	}
	target := specFields[rng.Intn(len(specFields))]
	putField(obj, target.Path, MutateValue(target.Type, rng))

	updated, err := c.Update(obj)
	if err != nil {
		if isServiceError(err) {
			stats.RejectedFaults.Add(1)
			return
		}
		if errors.Is(err, storage.ErrConflict) {
			stats.Conflicts.Add(1)
			return
		}
		stats.recordViolation(fmt.Sprintf("rmw update %s: %v", k.Name, err))
		return
	}

	// Re-read in the same version and verify the hidden fields survived.
	reread, err := c.Get(updated.Metadata.Namespace, updated.Metadata.Name)
	if err != nil {
		if isServiceError(err) {
			stats.RejectedFaults.Add(1)
			return
		}
		stats.recordViolation(fmt.Sprintf("rmw reread %s: %v", k.Name, err))
		return
	}
	unknownAfter := snapshotUnknown(k, version, reread)
	if !mapsEqual(unknownBefore, unknownAfter) {
		stats.recordViolation(fmt.Sprintf(
			"FIELD LOSS in %s/%s via %s: hidden fields changed by old-client RMW\nbefore=%v\nafter =%v",
			k.Name, obj.Key(), version, unknownBefore, unknownAfter))
		return
	}
	stats.Success.Add(1)
}

// snapshotUnknown captures the stash annotation(s) holding fields invisible
// to clientVersion — that is exactly the data an old client must not clobber.
func snapshotUnknown(k *schema.Kind, clientVersion string, obj *types.Object) map[string]string {
	out := map[string]string{}
	for key, val := range obj.Metadata.Annotations {
		if len(key) > len(types.StashAnnotationPrefix) &&
			key[:len(types.StashAnnotationPrefix)] == types.StashAnnotationPrefix {
			// only stashes belonging to versions at or older than the
			// client's represent fields the client cannot see/edit
			stashed := key[len(types.StashAnnotationPrefix):]
			if olderOrEqual(k, stashed, clientVersion) {
				out[key] = val
			}
		}
	}
	return out
}

func olderOrEqual(k *schema.Kind, a, b string) bool {
	ia, errA := k.VersionIndex(a)
	ib, errB := k.VersionIndex(b)
	if errA != nil || errB != nil {
		return false
	}
	return ia <= ib
}

func specOnly(fields []schema.FieldDef) []schema.FieldDef {
	var out []schema.FieldDef
	for _, f := range fields {
		if len(f.Path) > 5 && f.Path[:5] == "spec." {
			out = append(out, f)
		}
	}
	return out
}

func isServiceError(err error) bool {
	var down *convert.ErrServiceDown
	if errors.As(err, &down) {
		return true
	}
	var slow *convert.ErrSlowConversion
	return errors.As(err, &slow)
}

func mapsEqual(a, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if b[k] != v {
			return false
		}
	}
	return true
}

func lower(s string) string {
	b := []byte(s)
	for i := range b {
		if b[i] >= 'A' && b[i] <= 'Z' {
			b[i] += 'a' - 'A'
		}
	}
	return string(b)
}
