from wtc4d.corpus import registry


def test_load_registry_parses_all_entries():
    entries = registry.load_registry()
    assert len(entries) >= 10
    ids = [e.id for e in entries]
    assert len(ids) == len(set(ids)), "duplicate registry ids"
    assert "ia_911_tv_archive" in ids


def test_by_id_matches_load():
    by_id = registry.by_id()
    entries = registry.load_registry()
    assert set(by_id) == {e.id for e in entries}


def test_harvest_method_values_are_valid():
    for entry in registry.load_registry():
        assert entry.harvest_method in registry.HarvestMethod
