"""
KineticGuard - Local OpenSearch Persistent Project Memory & History Layer
Provides a persistent, zero-AWS, locally runnable OpenSearch storage engine.
Supports standard OpenSearch indexing, document retrieval, and DSL search queries.
If a live OpenSearch cluster is reachable at http://localhost:9200, it connects;
otherwise, it executes against a local persistent disk-backed index engine in output/opensearch_data/.
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import opensearchpy
    from opensearchpy import OpenSearch
    HAS_OPENSEARCH_PY = True
except ImportError:
    HAS_OPENSEARCH_PY = False


# Standard KineticGuard OpenSearch Indexes
INDEX_INCIDENTS = "kineticguard-incidents"
INDEX_WORKER_EXPOSURE = "kineticguard-worker-exposure"
INDEX_STATION_RISK = "kineticguard-station-risk"
INDEX_AGENT_DECISIONS = "kineticguard-agent-decisions"
INDEX_INTERVENTIONS = "kineticguard-interventions"
ALL_INDEXES = [
    INDEX_INCIDENTS,
    INDEX_WORKER_EXPOSURE,
    INDEX_STATION_RISK,
    INDEX_AGENT_DECISIONS,
    INDEX_INTERVENTIONS,
]


class LocalOpenSearchStorage:
    """
    Local disk-backed OpenSearch implementation.
    Persists documents and mappings to JSON on disk and executes query DSL.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir or "output/opensearch_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._indices: Dict[str, Dict[str, Any]] = {}
        self._init_storage()

    def _init_storage(self):
        for idx in ALL_INDEXES:
            idx_dir = self.data_dir / idx
            idx_dir.mkdir(parents=True, exist_ok=True)
            self._indices[idx] = {}
            # Load existing docs from disk
            for f in idx_dir.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        self._indices[idx][f.stem] = json.load(fp)
                except Exception:
                    pass

    def index(self, index: str, id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Indexes a document into the specified index and persists it to disk."""
        if index not in self._indices:
            self._indices[index] = {}
            (self.data_dir / index).mkdir(parents=True, exist_ok=True)

        doc = dict(body)
        if "_id" not in doc:
            doc["_id"] = id
        if "@timestamp" not in doc:
            doc["@timestamp"] = datetime.now(timezone.utc).isoformat()

        self._indices[index][id] = doc

        # Persist to disk
        safe_id = re.sub(r'[\\/*?:"<>|]', "_", str(id))
        doc_file = self.data_dir / index / f"{safe_id}.json"
        with open(doc_file, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)

        # Sync to cloud storage if GCS persistence configured
        try:
            from .cloud_storage import cloud_storage_manager
            if cloud_storage_manager.is_enabled:
                cloud_storage_manager.sync_to_cloud_async(doc_file)
        except Exception:
            pass

        return {
            "_index": index,
            "_id": id,
            "_version": 1,
            "result": "created",
            "_shards": {"total": 1, "successful": 1, "failed": 0},
        }

    def get(self, index: str, id: str) -> Dict[str, Any]:
        """Retrieves a document by index and ID."""
        if index not in self._indices or id not in self._indices[index]:
            raise KeyError(f"Document {id} not found in index {index}")
        doc = self._indices[index][id]
        return {
            "_index": index,
            "_id": id,
            "_source": doc,
            "found": True,
        }

    def search(self, index: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Executes a query against the specified index supporting match, term, and sort."""
        if index not in self._indices:
            return {"hits": {"total": {"value": 0, "relation": "eq"}, "hits": []}}

        all_docs = list(self._indices[index].values())
        body = body or {}
        query = body.get("query", {})
        size = body.get("size", 100)
        sort_cfg = body.get("sort", [])

        # Filter docs based on query DSL
        matched_docs = []
        for doc in all_docs:
            if self._matches_query(doc, query):
                matched_docs.append(doc)

        # Apply sorting if requested
        if sort_cfg:
            for s in sort_cfg:
                if isinstance(s, dict):
                    field, order = next(iter(s.items()))
                    is_desc = isinstance(order, dict) and order.get("order") == "desc" or order == "desc"
                    matched_docs.sort(key=lambda d: d.get(field, 0) or 0, reverse=is_desc)

        limited_docs = matched_docs[:size]
        hits = [{"_index": index, "_id": d.get("_id", ""), "_source": d} for d in limited_docs]

        return {
            "took": 1,
            "timed_out": False,
            "hits": {
                "total": {"value": len(matched_docs), "relation": "eq"},
                "hits": hits,
            },
        }

    def _matches_query(self, doc: Dict[str, Any], query: Dict[str, Any]) -> bool:
        if not query or "match_all" in query:
            return True

        # Handle 'term' query
        if "term" in query:
            for k, v in query["term"].items():
                expected = v.get("value") if isinstance(v, dict) else v
                val = self._resolve_nested(doc, k)
                if str(val).lower() != str(expected).lower():
                    return False
            return True

        # Handle 'match' query
        if "match" in query:
            for k, v in query["match"].items():
                expected = v.get("query") if isinstance(v, dict) else v
                val = self._resolve_nested(doc, k)
                if str(expected).lower() not in str(val).lower():
                    return False
            return True

        # Handle 'bool' query with 'must' / 'filter' / 'should' / 'must_not'
        if "bool" in query:
            bool_clause = query["bool"]
            if "must" in bool_clause:
                for must_clause in bool_clause.get("must", []):
                    if not self._matches_query(doc, must_clause):
                        return False
            if "filter" in bool_clause:
                for filter_clause in bool_clause.get("filter", []):
                    if not self._matches_query(doc, filter_clause):
                        return False
            if "must_not" in bool_clause:
                for not_clause in bool_clause.get("must_not", []):
                    if self._matches_query(doc, not_clause):
                        return False
            if "should" in bool_clause:
                should_clauses = bool_clause.get("should", [])
                if should_clauses and not any(self._matches_query(doc, s) for s in should_clauses):
                    return False
            return True

        return True

    def _resolve_nested(self, doc: Dict[str, Any], path: str) -> Any:
        parts = path.split(".")
        cur = doc
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return ""
        return cur


class KineticGuardOpenSearch:
    """
    Unified OpenSearch interface for KineticGuard project history.
    Provides automated fallbacks, schema initialization, and typed query methods.
    Supports remote OpenSearch clusters on Cloud Run and local persistent fallback.
    """
    _cluster_checked = False
    _cluster_available = False

    def __init__(self, host: Optional[str] = None, data_dir: Optional[Path] = None):
        self.host = host or os.getenv("OPENSEARCH_URL") or os.getenv("OPENSEARCH_HOST") or "http://localhost:9200"
        self.data_dir = Path(data_dir or "output/opensearch_data")
        self.live_client: Optional[Any] = None
        self.local_storage = LocalOpenSearchStorage(self.data_dir)
        self.is_live_cluster = False
        self._connect()

    def _connect(self):
        """Attempts to connect to live OpenSearch; falls back to local storage."""
        if not KineticGuardOpenSearch._cluster_checked:
            KineticGuardOpenSearch._cluster_checked = True
            if HAS_OPENSEARCH_PY:
                try:
                    kwargs = {"hosts": [self.host], "timeout": 1.0, "max_retries": 1}
                    u = os.getenv("OPENSEARCH_USER")
                    p = os.getenv("OPENSEARCH_PASSWORD")
                    if u and p:
                        kwargs["http_auth"] = (u, p)
                    if self.host.startswith("https://") or os.getenv("OPENSEARCH_USE_SSL", "false").lower() in ("true", "1"):
                        kwargs["use_ssl"] = True
                        kwargs["verify_certs"] = os.getenv("OPENSEARCH_VERIFY_CERTS", "true").lower() in ("true", "1")

                    client = OpenSearch(**kwargs)
                    info = client.info()
                    if info:
                        KineticGuardOpenSearch._cluster_available = True
                        self.live_client = client
                        self.is_live_cluster = True
                        self._init_live_indices()
                        return
                except Exception:
                    KineticGuardOpenSearch._cluster_available = False

        self.is_live_cluster = False

    def _init_live_indices(self):
        """Ensures all standard KineticGuard indexes exist on the live cluster."""
        if not self.live_client:
            return
        for idx in ALL_INDEXES:
            try:
                if not self.live_client.indices.exists(index=idx):
                    self.live_client.indices.create(index=idx)
            except Exception:
                pass

    def index_document(self, index: str, doc_id: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """Indexes a document in both local persistent storage and live cluster if available."""
        # Always write to local persistent storage for 100% reliable local guarantees
        res_local = self.local_storage.index(index=index, id=doc_id, body=document)

        # Also replicate to live cluster if connected
        if self.live_client:
            try:
                self.live_client.index(index=index, id=doc_id, body=document, refresh=True)
            except Exception:
                pass

        return res_local

    def get_document(self, index: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves document from live cluster or local storage."""
        if self.live_client:
            try:
                res = self.live_client.get(index=index, id=doc_id)
                return res.get("_source")
            except Exception:
                pass
        try:
            res = self.local_storage.get(index=index, id=doc_id)
            return res.get("_source")
        except KeyError:
            return None

    def search(self, index: str, query: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Searches documents matching the query DSL."""
        if self.live_client:
            try:
                res = self.live_client.search(index=index, body=query)
                hits = res.get("hits", {}).get("hits", [])
                return [h.get("_source", {}) for h in hits]
            except Exception:
                pass
        res = self.local_storage.search(index=index, body=query)
        hits = res.get("hits", {}).get("hits", [])
        return [h.get("_source", {}) for h in hits]

    # Specialized Domain Methods
    def index_incident(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        inc_id = incident_data.get("incident_id") or incident_data.get("event_id", f"INC-{int(time.time())}")
        doc = dict(incident_data)
        doc["privacy_guard"] = "ACTIVE"
        doc["visual_deidentified"] = True
        return self.index_document(INDEX_INCIDENTS, inc_id, doc)

    def index_worker_exposure(self, worker_id: str, exposure_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.index_document(INDEX_WORKER_EXPOSURE, worker_id, exposure_data)

    def index_station_risk(self, station_id: str, station_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.index_document(INDEX_STATION_RISK, station_id, station_data)

    def index_agent_decision(self, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        inc_id = decision_data.get("incident_id", f"DEC-{int(time.time())}")
        doc = dict(decision_data)
        doc["privacy_guard"] = "ACTIVE"
        doc["visual_deidentified"] = True
        return self.index_document(INDEX_AGENT_DECISIONS, f"decision_{inc_id}", doc)

    def index_intervention(self, intervention_data: Dict[str, Any]) -> Dict[str, Any]:
        int_id = intervention_data.get("intervention_id", f"INTV-{int(time.time())}")
        return self.index_document(INDEX_INTERVENTIONS, int_id, intervention_data)

    def get_worker_profile(self, worker_id: str) -> Optional[Dict[str, Any]]:
        return self.get_document(INDEX_WORKER_EXPOSURE, worker_id)

    def get_station_profile(self, station_id: str) -> Optional[Dict[str, Any]]:
        return self.get_document(INDEX_STATION_RISK, station_id)

    def get_interventions_for_worker(self, worker_id: str) -> List[Dict[str, Any]]:
        query = {
            "query": {"term": {"worker_id": worker_id}},
            "sort": [{"@timestamp": {"order": "desc"}}],
        }
        return self.search(INDEX_INTERVENTIONS, query)

    def get_recent_incidents_for_worker(self, worker_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        query = {
            "query": {"term": {"worker_identity.worker_id": worker_id}},
            "size": limit,
            "sort": [{"@timestamp": {"order": "desc"}}],
        }
        return self.search(INDEX_INCIDENTS, query)


# Global Singleton
opensearch_engine = KineticGuardOpenSearch()
