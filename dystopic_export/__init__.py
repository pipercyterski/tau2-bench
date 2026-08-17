"""Offline export tooling: tau2-bench domain -> Dystopic platform artifacts.

Nothing in this package runs inside the sandbox. It reads the vendored tau2
domain modules and emits the JSON documents that get POSTed to the platform:

    export_world.py     tau2 DB          -> World variant ``initial_state``
    export_ontology.py  DB + golden repl -> ``ledger_schema``
    export_schemas.py   tau2 tools       -> ``tools_schema`` (+ledger_read/adapter)
    export_scenarios.py tau2 tasks       -> scenario payloads

The single source of truth for entity/field naming is ``contract.py`` --
every exporter keys off it, so a rename lands in one place.
"""
