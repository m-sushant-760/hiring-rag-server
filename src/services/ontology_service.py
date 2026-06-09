"""
Ontology Service — Phase 8 + Phase B (OR-group matching)
=========================================================
In-memory directed skills graph using NetworkX (zero extra infrastructure).

The graph models relationships between skills so that when a JD says
"REST API development", candidates listing "FastAPI" or "Flask" are
correctly matched — something pure embedding similarity often misses
for specific tool ↔ category pairs.

Graph structure
---------------
  Nodes  : individual skills, tools, certifications, role types
  Edges  : typed directed relationships
    IS_A        "FastAPI" → "REST Framework" → "API Development"
    RELATED_TO  "Deep Learning" → "MLOps"
    ALIAS       "K8s" → "Kubernetes", "Kafka" → "Apache Kafka"

Public API
----------
  expand_query_terms(terms)          → augment JD terms with related skills
                                       (used for Pinecone query expansion only)
  extract_skills_from_text(t)        → scan text and return matched ontology nodes
  get_related_skills(skill)          → neighbourhood of a single skill
                                       (undirected — for query expansion only)
  get_candidate_coverage(skill)      → upward directed traversal — what does
                                       this skill imply knowledge of?
  get_jd_skill_variants(skill)       → downward 1-hop — what specific tools
                                       satisfy this JD requirement?
  skills_match_score(c, j)           → structured overlap score (0.0–1.0)
                                       using directed traversal (no sibling bleed)
  group_jd_skills_by_parent(skills)  → Strategy B: group JD siblings into
                                       OR-groups via shared ontology parent
  refine_groups_with_text(text, g)   → Strategy A: promote/demote group type
                                       using OR/AND signals in JD text
  score_skill_groups(cand, groups)   → group-aware score (satisfied / total)
  add_custom_skill(skill, par)       → extend ontology at runtime
"""

import re
from dataclasses import dataclass, field

import networkx as nx


# ---------------------------------------------------------------------------
# JDSkillGroup dataclass
# ---------------------------------------------------------------------------

@dataclass
class JDSkillGroup:
    """
    A logical cluster of JD skill requirements.

    group_type : "OR"  — candidate satisfies group by matching ANY one skill
                 "AND" — candidate must match ALL skills in the group
    label      : human-readable name (typically the shared ontology parent)
    skills     : the individual skill nodes in this group
    satisfied  : filled in after candidate scoring
    satisfied_by: which candidate skills satisfied the group
    """
    group_type:   str
    label:        str
    skills:       list[str]
    satisfied:    bool       = False
    satisfied_by: list[str]  = field(default_factory=list)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_ontology() -> nx.DiGraph:
    G = nx.DiGraph()

    def hier(parent: str, children: list[str], rel: str = "IS_A"):
        for child in children:
            G.add_edge(child, parent, type=rel)

    # ── API / Backend ───────────────────────────────────────────────────────
    hier("API Development", ["REST API", "GraphQL", "gRPC", "WebSocket API", "OpenAPI"])
    hier("REST Framework",  ["FastAPI", "Flask", "Django REST Framework",
                              "Express.js", "Spring Boot", "ASP.NET Core", "Rails API"])
    G.add_edge("REST Framework", "REST API",        type="IS_A")
    G.add_edge("REST API",       "API Development", type="IS_A")

    # ── Python ──────────────────────────────────────────────────────────────
    hier("Python", ["FastAPI", "Flask", "Django", "SQLAlchemy",
                    "Pydantic", "Celery", "NumPy", "Pandas", "Scikit-learn"])

    # ── Cloud & Infra ───────────────────────────────────────────────────────
    hier("Cloud Computing", ["AWS", "GCP", "Azure", "Fly.io", "Heroku", "DigitalOcean"])
    hier("AWS", ["AWS Lambda", "AWS EC2", "AWS S3", "AWS RDS",
                 "AWS ECS", "AWS SageMaker", "AWS Bedrock"])
    hier("Container Orchestration", ["Kubernetes", "K8s", "Docker Swarm", "Amazon EKS", "GKE"])
    G.add_edge("K8s",       "Kubernetes",              type="ALIAS")
    G.add_edge("Kubernetes","Container Orchestration", type="IS_A")
    hier("DevOps", ["Docker", "Kubernetes", "Terraform", "Ansible",
                    "GitHub Actions", "Jenkins", "CI/CD", "Helm"])

    # ── Data Engineering ────────────────────────────────────────────────────
    hier("Data Engineering", ["Apache Kafka", "Apache Spark", "Airflow",
                               "dbt", "Snowflake", "BigQuery", "Redshift",
                               "Databricks", "ETL", "ELT", "Data Pipelines"])
    G.add_edge("SQL", "Data Engineering", type="REQUIRED_FOR")
    hier("Database", ["PostgreSQL", "MySQL", "MongoDB", "Redis",
                      "Elasticsearch", "Cassandra", "DynamoDB", "SQLite"])

    # ── ML / AI ─────────────────────────────────────────────────────────────
    hier("Deep Learning",      ["PyTorch", "TensorFlow", "Keras", "JAX", "Hugging Face"])
    hier("Machine Learning",   ["Scikit-learn", "XGBoost", "LightGBM", "CatBoost",
                                 "Feature Engineering", "Model Evaluation"])
    G.add_edge("Deep Learning",  "Machine Learning", type="IS_A")
    G.add_edge("Deep Learning",  "MLOps",            type="RELATED_TO")
    hier("MLOps",              ["MLflow", "Weights & Biases", "Kubeflow",
                                 "BentoML", "Model Serving", "Model Monitoring"])
    hier("LLM / Generative AI",["LangChain", "LlamaIndex", "OpenAI API",
                                 "RAG", "Prompt Engineering", "Fine-tuning",
                                 "Vector Databases"])
    hier("Vector Databases",   ["Pinecone", "Weaviate", "Qdrant", "Chroma", "FAISS"])

    # ── Frontend ────────────────────────────────────────────────────────────
    hier("Frontend Development", ["React", "Vue.js", "Angular", "Next.js",
                                   "TypeScript", "JavaScript", "HTML", "CSS",
                                   "Tailwind CSS"])

    # ── Software Engineering ─────────────────────────────────────────────────
    hier("Software Engineering", ["System Design", "Microservices",
                                   "Event-Driven Architecture",
                                   "Domain-Driven Design",
                                   "Test-Driven Development",
                                   "Clean Code", "SOLID Principles"])
    G.add_edge("Microservices", "API Development", type="RELATED_TO")

    # ── Data Science ────────────────────────────────────────────────────────
    hier("Data Science", ["Statistical Analysis", "A/B Testing",
                           "Data Visualization", "Jupyter", "R", "MATLAB"])
    G.add_edge("Machine Learning", "Data Science", type="RELATED_TO")

    # ── Leadership / Process ────────────────────────────────────────────────
    hier("Agile",      ["Scrum", "Kanban", "Sprint Planning", "Retrospectives"])
    hier("Leadership", ["Team Lead", "Engineering Manager", "Technical Lead",
                         "Mentoring", "Hiring"])

    # ── Certifications → skill mapping ───────────────────────────────────────
    G.add_edge("AWS Solutions Architect",    "Cloud Computing", type="IS_A")
    G.add_edge("CKA",                        "Kubernetes",      type="IS_A")
    G.add_edge("CKAD",                       "Kubernetes",      type="IS_A")
    G.add_edge("PMP",                        "Leadership",      type="RELATED_TO")
    G.add_edge("Google Cloud Professional",  "GCP",            type="IS_A")

    # ── T1: Common shorthands / aliases ──────────────────────────────────────
    # Only added for unambiguous shorthands — i.e. "Kafka" can only mean
    # Apache Kafka in a tech context.  Each ALIAS edge lets extract_skills_from_text
    # pick up the shorthand and directed traversal follow it to the canonical node.
    G.add_edge("Kafka",      "Apache Kafka",  type="ALIAS")
    G.add_edge("Spark",      "Apache Spark",  type="ALIAS")
    G.add_edge("HuggingFace","Hugging Face",  type="ALIAS")
    G.add_edge("Langchain",  "LangChain",     type="ALIAS")
    G.add_edge("Llamaindex", "LlamaIndex",    type="ALIAS")
    G.add_edge("Postgres",   "PostgreSQL",    type="ALIAS")
    G.add_edge("Mongo",      "MongoDB",       type="ALIAS")
    G.add_edge("Dynamo",     "DynamoDB",      type="ALIAS")
    G.add_edge("sklearn",    "Scikit-learn",  type="ALIAS")
    G.add_edge("scikit",     "Scikit-learn",  type="ALIAS")

    return G


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_graph: nx.DiGraph | None = None


def get_graph() -> nx.DiGraph:
    global _graph
    if _graph is None:
        _graph = _build_ontology()
    return _graph


# ---------------------------------------------------------------------------
# Public API — Query Expansion (intentionally undirected)
# ---------------------------------------------------------------------------

def get_related_skills(skill: str, max_hops: int = 2) -> set[str]:
    """
    Return all skills within *max_hops* graph edges from *skill*.

    Uses undirected BFS intentionally — this function is used ONLY for
    Pinecone query expansion, where broad recall is the goal.  The wider
    neighbourhood (parents, children, siblings) produces a richer query.

    Do NOT use this for scoring.  Use get_candidate_coverage() and
    get_jd_skill_variants() for score-critical traversal.
    """
    G = get_graph()
    if skill not in G:
        return {skill}
    UG = G.to_undirected()
    return set(nx.single_source_shortest_path(UG, skill, cutoff=max_hops))


def expand_query_terms(terms: list[str], max_hops: int = 1) -> set[str]:
    """
    Expand a list of JD terms using the ontology graph.

    For each term, adds parent categories, child technologies, and
    related concepts within *max_hops* edges.

    Example:
        expand_query_terms(["REST API"])
        → {"REST API", "API Development", "REST Framework",
           "FastAPI", "Flask", "Django REST Framework", ...}
    """
    G = get_graph()
    expanded: set[str] = set(terms)
    for term in terms:
        # Exact match
        if term in G:
            expanded.update(get_related_skills(term, max_hops=max_hops))
            continue
        # Case-insensitive match
        term_lower = term.lower()
        for node in G.nodes():
            if node.lower() == term_lower:
                expanded.update(get_related_skills(node, max_hops=max_hops))
                break
        # Substring match for multi-word JD phrases like "REST API development"
        for node in G.nodes():
            if node.lower() in term_lower or term_lower in node.lower():
                expanded.update(get_related_skills(node, max_hops=1))
    return expanded


def extract_skills_from_text(
    text: str,
    deduplicate_aliases: bool = False,
) -> list[str]:
    """
    Return all ontology skill nodes found (case-insensitive) in *text*.

    deduplicate_aliases (default False):
      When True, drops ALIAS source nodes (e.g. 'Kafka', 'Mongo', 'Dynamo')
      if their canonical target ('Apache Kafka', 'MongoDB', 'DynamoDB') is
      also detected in the same text.  Pass True when scanning JD text where
      canonical names are expected.

      Leave False for resume scanning — shorthands like 'Kafka' should be
      kept so that get_candidate_coverage('Kafka') can traverse upward to
      'Apache Kafka' and credit the JD requirement correctly.

    Rule: drop ALIAS source A when canonical(A) is also detected.
    This preserves K8s when 'kubernetes' is NOT in the text (no collision),
    and drops 'Mongo' when 'mongodb' IS in the text (substring collision).
    """
    G = get_graph()
    text_lower = text.lower()
    detected = [node for node in G.nodes() if node.lower() in text_lower]
    if not deduplicate_aliases:
        return detected
    # Build alias source → canonical target map for O(1) lookup
    alias_to_canonical: dict[str, str] = {
        u: v
        for u, v, data in G.edges(data=True)
        if data.get("type") == "ALIAS"
    }
    detected_set = set(detected)
    return [
        node for node in detected
        if node not in alias_to_canonical            # not an alias source → keep
        or alias_to_canonical[node] not in detected_set  # canonical absent → keep
    ]


# ---------------------------------------------------------------------------
# Public API — Directed Traversal Helpers (T2)
# ---------------------------------------------------------------------------

def get_candidate_coverage(candidate_skill: str, max_hops: int = 2) -> set[str]:
    """
    Return the set of JD requirements this candidate skill covers.

    Traverses UPWARD only (child → parent, following IS_A / ALIAS edges in
    the direction they point).  This models the correct semantic:
      "A candidate who knows FastAPI implicitly covers REST Framework, REST API,
       API Development, and Python."
    But NOT:
      "A candidate who knows REST API therefore knows Flask."  ← downward, wrong

    ALIAS edges are upward too:
      "Kafka" → "Apache Kafka" → "Data Engineering"

    Sibling skills (e.g. Azure and AWS, both children of Cloud Computing) are
    never reachable from each other via upward traversal.  This prevents the
    false-positive sibling-credit bug.

    Examples:
      get_candidate_coverage("FastAPI")  → {FastAPI, REST Framework, REST API,
                                            API Development, Python}
      get_candidate_coverage("Azure")    → {Azure, Cloud Computing}
      get_candidate_coverage("K8s")      → {K8s, Kubernetes, Container Orchestration,
                                            DevOps}
      get_candidate_coverage("Kafka")    → {Kafka, Apache Kafka, Data Engineering}
    """
    G = get_graph()
    if candidate_skill not in G:
        return {candidate_skill}
    covered: set[str] = {candidate_skill}
    frontier: set[str] = {candidate_skill}
    for _ in range(max_hops):
        next_frontier: set[str] = set()
        for node in frontier:
            next_frontier.update(G.successors(node))   # upward: child→parent direction
        new_nodes = next_frontier - covered
        if not new_nodes:
            break
        covered.update(new_nodes)
        frontier = new_nodes
    return covered


def get_jd_skill_variants(jd_skill: str) -> set[str]:
    """
    Return the set of specific tools/skills that directly satisfy a JD requirement.

    Traverses DOWNWARD only (parent → direct children, i.e. predecessors in the
    directed graph).  Limited to 1 hop intentionally — we only credit direct
    implementations of the requirement, not grandchildren or cousins.

    Examples:
      get_jd_skill_variants("REST API")      → {REST API, FastAPI, Flask,
                                                Django REST Framework, ...}
      get_jd_skill_variants("Kubernetes")    → {Kubernetes, K8s, CKA, CKAD}
      get_jd_skill_variants("Apache Kafka")  → {Apache Kafka, Kafka}
      get_jd_skill_variants("AWS")           → {AWS, AWS Lambda, AWS EC2,
                                                AWS S3, AWS RDS, ...}

    NOTE: "AWS" as a JD skill accepts its own service children (Lambda, EC2…)
    but NOT sibling clouds (Azure, GCP) — they are under Cloud Computing,
    not under AWS.
    """
    G = get_graph()
    variants: set[str] = {jd_skill}
    if jd_skill in G:
        variants.update(G.predecessors(jd_skill))   # downward: parent→child direction
    return variants


# ---------------------------------------------------------------------------
# Public API — Flat Skill Scoring (T3, kept for backward compat)
# ---------------------------------------------------------------------------

def skills_match_score(
    candidate_skills: list[str],
    jd_skills: list[str],
    max_hops: int = 2,
) -> float:
    """
    Structured skill match score (0.0–1.0) using directed ontology traversal.

    Replaced the old undirected BFS implementation to eliminate sibling false
    positives (Azure crediting AWS, Vue.js crediting React, etc.).

    Matching rule per JD skill:
      For each JD skill, build the set of tools that directly satisfy it
      (get_jd_skill_variants — 1-hop downward).  For each candidate skill,
      build what it implies upward (get_candidate_coverage — max_hops upward).
      Credit the JD skill if any candidate coverage overlaps any JD variant.

    What this prevents vs. old implementation:
      - Azure does NOT credit AWS (siblings under Cloud Computing) ✅
      - Vue.js does NOT credit React (siblings under Frontend Development) ✅
      - Apache Spark does NOT credit Apache Kafka (siblings under Data Engineering) ✅

    What is correctly preserved:
      - FastAPI covers REST Framework and REST API (upward 2 hops) ✅
      - K8s covers Kubernetes (ALIAS, 1 hop) ✅
      - Kafka covers Apache Kafka (ALIAS via T1 edge, 1 hop) ✅

    Note: prefer score_skill_groups() for new code — it handles OR-grouped JD
    requirements correctly and produces richer output.
    """
    if not jd_skills:
        return 0.0
    matched = 0
    for jd_skill in jd_skills:
        jd_variants_lower = {v.lower() for v in get_jd_skill_variants(jd_skill)}
        for cand_skill in candidate_skills:
            cand_coverage_lower = {c.lower() for c in get_candidate_coverage(cand_skill, max_hops)}
            if cand_coverage_lower & jd_variants_lower:
                matched += 1
                break   # JD skill is covered — move to next JD skill
    return matched / len(jd_skills)


# ---------------------------------------------------------------------------
# Public API — OR-Group Detection (T5: Strategy B + A)
# ---------------------------------------------------------------------------

def group_jd_skills_by_parent(jd_skills: list[str]) -> list[JDSkillGroup]:
    """
    Strategy B — Group JD skills that share a direct ontology parent into OR-groups.

    Rationale: siblings in the ontology (e.g. AWS, GCP, Azure all under
    Cloud Computing) represent alternative implementations of the same
    category.  When a JD lists 2+ siblings, the candidate should be credited
    for the entire category by satisfying any one of them.

    Algorithm:
      1. For each JD skill, find its direct parents (G.successors, upward).
      2. Build parent → [jd_skills that are children] map.
      3. Parents with 2+ JD-skill children → candidate OR-group.
      4. Filter: remove parent-child pairs within the same group.
         (AWS and AWS Lambda should not be OR-alternatives — Lambda IS_A AWS.)
      5. Deduplicate identical skill sets — keep the most specific parent
         (fewest total children in the graph).
      6. Skills not in any multi-member group → singleton AND-requirements.

    Examples:
      jd_skills = ["AWS", "GCP", "Azure", "Python", "FastAPI", "Flask",
                   "Docker", "Kubernetes"]
      →  [
           JDSkillGroup(OR,  "Cloud Computing", ["AWS", "GCP", "Azure"]),
           JDSkillGroup(OR,  "Python",          ["FastAPI", "Flask"]),
           JDSkillGroup(AND, "Python",          ["Python"]),
           JDSkillGroup(AND, "Docker",          ["Docker"]),
           JDSkillGroup(AND, "Kubernetes",      ["Kubernetes"]),
         ]
    """
    G = get_graph()

    # Step 1 & 2: map parent → jd_skill children
    parent_to_children: dict[str, list[str]] = {}
    for skill in jd_skills:
        if skill not in G:
            continue
        for parent in G.successors(skill):
            parent_to_children.setdefault(parent, []).append(skill)

    # Step 3: collect candidate OR-groups (2+ JD-skill children under same parent)
    candidate_groups: list[tuple[str, list[str]]] = []
    for parent, children in parent_to_children.items():
        if len(children) < 2:
            continue
        # Step 4: filter out skills that are ancestors of other skills in the group.
        # Example: if both "AWS" and "AWS Lambda" are in children, AWS is an
        # ancestor of Lambda — they should not be OR-alternatives.
        filtered: list[str] = []
        for skill in children:
            is_ancestor_of_another = any(
                skill in get_candidate_coverage(other, max_hops=3)
                for other in children
                if other != skill
            )
            if not is_ancestor_of_another:
                filtered.append(skill)
        if len(filtered) >= 2:
            candidate_groups.append((parent, sorted(filtered)))

    # Step 5: deduplicate groups with identical skill sets.
    # If Python and REST Framework both produce {FastAPI, Flask}, keep the
    # most specific parent (fewest total direct children in the full graph).
    seen: dict[frozenset, tuple[str, list[str]]] = {}
    for parent, skills in candidate_groups:
        key = frozenset(skills)
        if key not in seen:
            seen[key] = (parent, skills)
        else:
            existing_parent = seen[key][0]
            existing_count  = len(list(G.predecessors(existing_parent)))
            current_count   = len(list(G.predecessors(parent)))
            if current_count < existing_count:
                seen[key] = (parent, skills)

    # Collect all skills absorbed into multi-member OR groups
    skills_in_groups: set[str] = {s for fs in seen for s in fs}

    # Step 6: build final list
    groups: list[JDSkillGroup] = []

    # OR-groups (Strategy B)
    for parent, skills in seen.values():
        groups.append(JDSkillGroup(group_type="OR", label=parent, skills=skills))

    # Labels already used as the parent identifier for a multi-member group.
    # Example: if "Data Engineering" is the label of a multi-skill group, do
    # NOT also emit it as a standalone singleton — it would double-count the
    # same requirement in the score denominator.
    already_used_labels: set[str] = {parent for parent, _ in seen.values()}

    # Singletons — skills not placed in any OR-group become AND-requirements,
    # unless the skill is already a label for a multi-member group above.
    for skill in jd_skills:
        if skill not in skills_in_groups and skill not in already_used_labels:
            groups.append(JDSkillGroup(group_type="AND", label=skill, skills=[skill]))

    return groups


# Strategy A regex patterns
# OR signals: confirm or upgrade a group to OR
_OR_SIGNALS = re.compile(
    # Matches explicit alternation words/symbols:
    #   \bor\b, \beither\b, \band/or\b  — standard alternation keywords
    #   (?<!\w)/(?!\w)                  — standalone slash separator (A / B)
    #                                     but NOT slash inside tokens like CI/CD
    r"(?:\bor\b|\beither\b|\band/or\b|(?<!\w)/(?!\w))",
    re.IGNORECASE,
)

# AND signals: downgrade a group to AND (only when no OR signal also present)
_AND_SIGNALS = re.compile(
    r"\b(?:and|as well as|along with|both|plus)\b",
    re.IGNORECASE,
)

# Small buffer (chars) added to each side of the inter-skill span.
# Kept tight (20) so Strategy A only captures conjunctions immediately
# adjacent to the skill cluster — prevents sweeping into unrelated
# adjacent JD lines that contain their own OR/AND signals.
_WINDOW_BUFFER = 20


def refine_groups_with_text(
    jd_text: str,
    groups: list[JDSkillGroup],
) -> list[JDSkillGroup]:
    """
    Strategy A — Refine group types by scanning the JD text for OR/AND signals
    within each group's skill span.

    OR signals ("or", "either", "/", "and/or"):
      → confirm or upgrade group_type to "OR"
    AND signals ("and", "as well as", "both", "along with"):
      → downgrade group_type to "AND" (only when no OR signal also found)

    Only multi-skill groups (2+ skills) are examined; singletons remain AND.

    When both signals appear in the same window, OR takes precedence — this
    errs toward leniency when the JD itself is ambiguous.

    Window strategy: scan from the first to the last skill position, plus a
    small ±_WINDOW_BUFFER (20 chars) on each side.  This deliberately avoids
    the adjacent JD lines — keeping the window within the skill cluster prevents
    false OR signals from neighbouring bullet points (e.g. "FastAPI, or Django"
    contaminating the DevOps group window).

    Examples:
      "familiarity with AWS, GCP, or Azure"         → OR (confirmed)
      "experience across AWS and Azure"              → AND (downgraded)
      "Flask, FastAPI, or Django frameworks"         → OR (confirmed)
      "you will use Kafka, Spark, and Airflow daily" → AND (downgraded)
      "Docker and Kubernetes" + "Jenkins, GitHub Actions"  → AND (correctly)
    """
    text_lower = jd_text.lower()

    for group in groups:
        if len(group.skills) < 2:
            continue   # singletons are unambiguously AND — nothing to refine

        # Find first occurrence position of each skill in the JD text,
        # and also record the end-position of the last skill found.
        skill_spans: list[tuple[int, int]] = []   # (start, end) per skill
        for skill in group.skills:
            idx = text_lower.find(skill.lower())
            if idx >= 0:
                skill_spans.append((idx, idx + len(skill)))

        if len(skill_spans) < 2:
            continue   # fewer than 2 skills found in text — keep Strategy B default

        # Scan only the inter-skill span: from the start of the first skill
        # to the end of the last skill, padded by _WINDOW_BUFFER on each side.
        # This keeps the window within the skill cluster and prevents adjacent
        # lines from injecting false OR/AND signals.
        span_start = min(s for s, _ in skill_spans)
        span_end   = max(e for _, e in skill_spans)
        window_start = max(0, span_start - _WINDOW_BUFFER)
        window_end   = min(len(jd_text), span_end + _WINDOW_BUFFER)
        window       = jd_text[window_start:window_end]

        or_found  = bool(_OR_SIGNALS.search(window))
        and_found = bool(_AND_SIGNALS.search(window))

        if or_found:
            group.group_type = "OR"      # OR signal wins outright
        elif and_found:
            group.group_type = "AND"     # AND-only signal → downgrade
        # Neither found → keep Strategy B default (OR for multi-member groups)

    return groups


# ---------------------------------------------------------------------------
# Public API — Group-Aware Scoring (T5)
# ---------------------------------------------------------------------------

def score_skill_groups(
    candidate_skills: list[str],
    groups: list[JDSkillGroup],
    max_hops: int = 2,
) -> tuple[float, list[JDSkillGroup]]:
    """
    Score a candidate against a list of JDSkillGroups.

    Returns:
        (score_0_to_1, annotated_groups)

    score = satisfied_groups / total_groups

    Group satisfaction:
      OR-group:  ANY group skill is covered by ANY candidate skill (upward traversal)
      AND-group: ALL group skills are covered by candidate skills

    The returned groups list has `satisfied` and `satisfied_by` fields filled in.

    Why this is correct:
      - JD says "AWS, GCP, or Azure" → OR-group {AWS, GCP, Azure}
        Candidate has Azure → Azure IS a member → group satisfied ✅
        (Not because Azure ≈ AWS, but because the JD declared them as alternatives)
      - JD says "AWS and GCP" (multi-cloud) → AND-group {AWS, GCP}
        Candidate has Azure only → NOT satisfied ❌ (correctly strict)
    """
    if not groups:
        return 0.0, []

    # Pre-compute each candidate skill's upward coverage (lower-cased for matching)
    cand_coverage_map: dict[str, set[str]] = {
        cs: {c.lower() for c in get_candidate_coverage(cs, max_hops)}
        for cs in candidate_skills
    }
    all_cand_coverage: set[str] = (
        set().union(*cand_coverage_map.values()) if cand_coverage_map else set()
    )

    satisfied_count = 0
    annotated: list[JDSkillGroup] = []

    for group in groups:
        # For each group skill, find its direct variants (downward 1-hop)
        group_variants: dict[str, set[str]] = {
            gs: {v.lower() for v in get_jd_skill_variants(gs)}
            for gs in group.skills
        }

        if group.group_type == "OR":
            # ANY one group skill satisfied by ANY candidate skill
            satisfied_by: list[str] = []
            for gs, variants in group_variants.items():
                if variants & all_cand_coverage:
                    contributing = [
                        cs for cs, cov in cand_coverage_map.items()
                        if variants & cov
                    ]
                    satisfied_by.extend(contributing)
            is_satisfied = len(satisfied_by) > 0

        else:   # AND-group
            # ALL group skills must be covered
            satisfied_by = []
            is_satisfied = True
            for gs, variants in group_variants.items():
                if variants & all_cand_coverage:
                    contributing = [
                        cs for cs, cov in cand_coverage_map.items()
                        if variants & cov
                    ]
                    satisfied_by.extend(contributing)
                else:
                    is_satisfied = False
                    break

        if is_satisfied:
            satisfied_count += 1

        annotated.append(JDSkillGroup(
            group_type   = group.group_type,
            label        = group.label,
            skills       = group.skills,
            satisfied    = is_satisfied,
            satisfied_by = sorted(set(satisfied_by)),
        ))

    score = satisfied_count / len(groups)
    return score, annotated


# ---------------------------------------------------------------------------
# Public API — Runtime extension
# ---------------------------------------------------------------------------

def add_custom_skill(skill: str, parent: str, relation: str = "IS_A") -> None:
    """Add a domain-specific skill to the ontology at runtime."""
    get_graph().add_edge(skill, parent, type=relation)
