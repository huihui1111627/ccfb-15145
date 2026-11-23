// Package storage persists custom-resource objects in each kind's configured
// storage version. All objects are held in their storage-version shape;
// reads and writes from any other version are converted by the conversion
// service before they touch stored bytes.
//
// Invariant: a record is either fully written (conversion succeeded before
// the write) or not written at all. If the conversion service fails or is
// too slow, the mutating call returns an error and storage stays untouched.
package storage

import (
	"errors"
	"fmt"
	"sort"
	"sync"

	"crv/pkg/convert"
	"crv/pkg/schema"
	"crv/pkg/types"
)

// ErrConflict is returned on an optimistic-concurrency mismatch.
var ErrConflict = errors.New("resource version conflict: object was modified concurrently")

// ErrNotFound is returned when no object matches the key.
var ErrNotFound = errors.New("object not found")

// record is the on-disk (here: in-memory) envelope. Only storage-version
// objects are ever placed in Data; all writes go through Encode so the
// envelope models real serialized bytes.
type record struct {
	StorageVersion string
	Data           []byte
}

// Store is the versioned object store.
type Store struct {
	reg *schema.Registry
	svc *convert.Service

	mu   sync.RWMutex
	data map[string]map[string]*record // kind -> key -> record
	rv   map[string]int64              // kind -> next resource version
}

func New(reg *schema.Registry, svc *convert.Service) *Store {
	return &Store{
		reg:  reg,
		svc:  svc,
		data: map[string]map[string]*record{},
		rv:   map[string]int64{},
	}
}

func (s *Store) storageVersion(kind string) (string, error) {
	k, err := s.reg.Get(kind)
	if err != nil {
		return "", err
	}
	return k.StorageVersion, nil
}

// Create stores a new object. The object may be in any served version; it
// is converted to the storage version first. On conversion failure nothing
// is written and the input object is left unchanged.
func (s *Store) Create(obj *types.Object) (*types.Object, error) {
	sv, err := s.storageVersion(obj.Kind)
	if err != nil {
		return nil, err
	}
	stored := obj.DeepCopy()
	if err := s.svc.ConvertObject(stored, sv); err != nil {
		return nil, err // explicit failure; no partial data
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	bucket, ok := s.data[obj.Kind]
	if !ok {
		bucket = map[string]*record{}
		s.data[obj.Kind] = bucket
	}
	if _, exists := bucket[obj.Key()]; exists {
		return nil, fmt.Errorf("object %s already exists", obj.Key())
	}
	s.rv[obj.Kind]++
	stored.Metadata.ResourceVersion = s.rv[obj.Kind]
	raw, err := types.Encode(stored)
	if err != nil {
		return nil, err
	}
	bucket[obj.Key()] = &record{StorageVersion: sv, Data: raw}
	return stored.DeepCopy(), nil
}

// Get reads an object and converts it from storage to wantVersion.
func (s *Store) Get(kind, namespace, name, wantVersion string) (*types.Object, error) {
	s.mu.RLock()
	rec, ok := s.data[kind][namespace+"/"+name]
	s.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("%w: %s/%s", ErrNotFound, namespace, name)
	}
	obj, err := types.Decode(rec.Data)
	if err != nil {
		return nil, err
	}
	if err := s.svc.ConvertObject(obj, wantVersion); err != nil {
		return nil, err
	}
	return obj, nil
}

// Update applies a read-modify-write. The caller's expectedRV must match the
// current resource version. The object is converted to the storage version
// before the CAS; a conversion failure aborts before any state change.
func (s *Store) Update(obj *types.Object, expectedRV int64) (*types.Object, error) {
	sv, err := s.storageVersion(obj.Kind)
	if err != nil {
		return nil, err
	}
	stored := obj.DeepCopy()
	if err := s.svc.ConvertObject(stored, sv); err != nil {
		return nil, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	bucket := s.data[obj.Kind]
	rec, ok := bucket[obj.Key()]
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrNotFound, obj.Key())
	}
	current, err := types.Decode(rec.Data)
	if err != nil {
		return nil, err
	}
	if current.Metadata.ResourceVersion != expectedRV {
		return nil, ErrConflict
	}
	s.rv[obj.Kind]++
	stored.Metadata.ResourceVersion = s.rv[obj.Kind]
	raw, err := types.Encode(stored)
	if err != nil {
		return nil, err
	}
	bucket[obj.Key()] = &record{StorageVersion: sv, Data: raw}
	return stored.DeepCopy(), nil
}

// List returns all objects of a kind converted to wantVersion, sorted by
// key for deterministic ordering.
func (s *Store) List(kind, wantVersion string) ([]*types.Object, error) {
	s.mu.RLock()
	recs := make([]*record, 0, len(s.data[kind]))
	for _, rec := range s.data[kind] {
		recs = append(recs, rec)
	}
	s.mu.RUnlock()
	sort.Slice(recs, func(i, j int) bool {
		ai, _ := types.Decode(recs[i].Data)
		aj, _ := types.Decode(recs[j].Data)
		return ai.Key() < aj.Key()
	})
	out := make([]*types.Object, 0, len(recs))
	for _, rec := range recs {
		obj, err := types.Decode(rec.Data)
		if err != nil {
			return nil, err
		}
		if err := s.svc.ConvertObject(obj, wantVersion); err != nil {
			return nil, err
		}
		out = append(out, obj)
	}
	return out, nil
}

// CountByStorageVersion reports how many stored objects of a kind still
// live in each storage-version shape — the migrator's progress signal.
func (s *Store) CountByStorageVersion(kind string) map[string]int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := map[string]int{}
	for _, rec := range s.data[kind] {
		out[rec.StorageVersion]++
	}
	return out
}

// Count returns the total number of stored objects of a kind.
func (s *Store) Count(kind string) int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.data[kind])
}

// MigrateOne is the primitive used by the background migrator. It reads one
// object still shaped like oldVersion, re-converts it to the current
// storage version and rewrites it under an optimistic CAS, so a concurrent
// client write can never be silently overwritten. Returns:
//   - migrated=true when this call rewrote the record
//   - migrated=false when the record was already current (another worker or
//     a client write got there first)
//   - error when the conversion service fails; the record stays untouched
func (s *Store) MigrateOne(kind, namespace, name, oldVersion string) (bool, error) {
	s.mu.Lock()
	rec, ok := s.data[kind][namespace+"/"+name]
	if !ok {
		s.mu.Unlock()
		return false, fmt.Errorf("%w: %s/%s", ErrNotFound, namespace, name)
	}
	// Re-check under the lock: a client write between scheduling and now
	// may already have rewritten the object in the new storage version.
	if rec.StorageVersion != oldVersion {
		s.mu.Unlock()
		return false, nil
	}
	current, err := types.Decode(rec.Data)
	if err != nil {
		s.mu.Unlock()
		return false, err
	}
	expectedRV := current.Metadata.ResourceVersion
	sv, err := s.storageVersion(kind)
	if err != nil {
		s.mu.Unlock()
		return false, err
	}
	s.mu.Unlock()

	// Conversion happens outside the store lock; on failure the record is
	// never touched.
	rewritten := current.DeepCopy()
	if err := s.svc.ConvertObject(rewritten, sv); err != nil {
		return false, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	bucket := s.data[kind]
	fresh, ok := bucket[namespace+"/"+name]
	if !ok {
		return false, fmt.Errorf("%w: %s/%s", ErrNotFound, namespace, name)
	}
	if fresh.StorageVersion != oldVersion {
		return false, nil // raced: already migrated by someone else
	}
	live, err := types.Decode(fresh.Data)
	if err != nil {
		return false, err
	}
	if live.Metadata.ResourceVersion != expectedRV {
		return false, nil // raced: client write wins, object is already new-version
	}
	s.rv[kind]++
	rewritten.Metadata.ResourceVersion = s.rv[kind]
	raw, err := types.Encode(rewritten)
	if err != nil {
		return false, err
	}
	bucket[namespace+"/"+name] = &record{StorageVersion: sv, Data: raw}
	return true, nil
}

// KeysForVersion lists object keys whose stored shape equals version.
func (s *Store) KeysForVersion(kind, version string) []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var out []string
	for key, rec := range s.data[kind] {
		if rec.StorageVersion == version {
			out = append(out, key)
		}
	}
	sort.Strings(out)
	return out
}

// RawRecord exposes the stored envelope for tests and the UI object view.
func (s *Store) RawRecord(kind, namespace, name string) (storageVersion string, obj *types.Object, ok bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	rec, exists := s.data[kind][namespace+"/"+name]
	if !exists {
		return "", nil, false
	}
	o, err := types.Decode(rec.Data)
	if err != nil {
		return "", nil, false
	}
	return rec.StorageVersion, o, true
}
