"""Trace format adapters, one subpackage per vendor format.

Each adapter reads its format into Parquet tables (see ``cta.table_schemas``
for the shared ones). ``cta.plugins.ADAPTERS`` lists the built-in adapters
and imports them on first use.
"""
