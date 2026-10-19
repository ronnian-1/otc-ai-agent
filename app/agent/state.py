"""Small, conservative rules for user-declared shopping context, not SKU facts.

Literal SKU IDs and explicit shopping phrases are parsed. A pending name is
bound separately by the program after a verified search, never by model prose.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


SKU_PATTERN = r"(?<![a-zA-Z0-9])S\d{3,}(?![a-zA-Z0-9])"
FIELD_TERMS = {
    "price": ("价格", "多少钱", "售价"),
    "spec": ("规格", "容量", "多少克", "多少粒"),
    "suitable_crowd": ("适用人群", "适合谁", "适合什么人"),
    "taboo": ("禁忌", "注意事项"),
    "delivery": ("发货", "配送", "多久到"),
    "stock": ("库存", "有货"),
}


def normalize_condition(text: str) -> str:
    """Normalize explicit scene wording only, without inferring health needs."""
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(?:这个条件|这一条件|的条件|条件|这个场景|的场景|场景)$", "", text)
    return re.sub(r"^(?:经常|常常)", "", text).strip(" ：:")


def _extract_name_query(text: str) -> str | None:
    """Strip query wrappers at the edges, preserving the product name itself.

    This is a literal name lookup, not semantic intent recognition. Statements
    with extra conditions/clauses are left unresolved rather than guessed.
    """
    text = text.strip(" \t\r\n，,。！？!?；;")
    text = re.sub(r"^(?:(?:请问|麻烦|帮我|请|查询|查一下|看一下|改看|换成|切换到|我说的是)\s*)+", "", text)
    terms = sorted((t for values in FIELD_TERMS.values() for t in values), key=len, reverse=True)
    fields = "(?:" + "|".join(re.escape(t) for t in terms) + ")"
    text = re.sub(rf"(?:的)?{fields}(?:(?:和|及|、){fields})*(?:是|为)?(?:多少|什么)?(?:呢|吗|呀|啊)?$", "", text)
    text = re.sub(r"呢$", "", text).strip()
    if not text or re.search(r"预算|目标|需求|不要|不是|比较|对比|[，,。！？!?；;]", text):
        return None
    return text


@dataclass
class SessionState:
    """Per-session declarations and rule evidence; never verified product data."""

    current_sku: str | None = None
    comparison_skus: tuple[str, ...] = ()
    budget_max: float | None = None
    goal: str | None = None
    excluded_directions: tuple[str, ...] = ()
    query_field: str | None = None
    sources: dict[str, dict[str, str]] = field(default_factory=dict)
    clarification: str | None = None
    name_query: str | None = None
    conditions: tuple[str, ...] = ()
    primary_condition: str | None = None
    pending_conditions: tuple[str, ...] = ()
    condition_changes: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def _source(self, field_name: str, text: str, basis: str = "user_explicit") -> None:
        self.sources[field_name] = {"text": text, "basis": basis}

    def _clear_product(self) -> None:
        self.current_sku = None
        self.comparison_skus = ()
        self.query_field = None
        for key in ("current_sku", "comparison_skus", "query_field"):
            self.sources.pop(key, None)

    def _update_conditions(self, text: str) -> None:
        """Bounded explicit edits, separate from product facts and medical history.

        Refocusing retains secondary scenes. Only named retractions remove them;
        unresolved references ask for clarification instead of choosing a target.
        The last edit summary is transient; sources retain the latest declaration.
        """
        before = self.conditions
        added, removed, pending = [], [], []
        focused = None
        explicit_edit = False
        for clause in re.split(r"[，,。！？!?；;]+", text):
            clause = clause.strip()
            if not clause:
                continue
            retract = re.search(
                r"(?:先)?(?:不要(?:再)?(?:考虑|关注)|不(?:再)?(?:考虑|关注)|取消|撤销|去掉)\s*(.+)", clause,
            )
            focus = re.search(r"(?:重点考虑|重新考虑|(?:更|主要(?:想)?)?(?:关注|考虑的场景是))\s*(.*)", clause)
            declaration = re.fullmatch(r"我(?:经常|常常)(.+)", clause)
            if retract:
                target = normalize_condition(retract[1])
                if target in self.conditions:
                    explicit_edit = True
                    self.conditions = tuple(c for c in self.conditions if c != target)
                    removed.append(target)
                    self._source("condition:" + target, clause, "user_retraction")
                    self._source("conditions", clause, "user_retraction")
                    if self.primary_condition == target:
                        self.primary_condition = self.conditions[0] if len(self.conditions) == 1 else None
                elif ("条件" in clause or "关注" in clause or "考虑" in clause):
                    pending.append(clause)
            elif focus or declaration:
                target = normalize_condition((focus or declaration)[1])
                # No guessing a demonstrative, alternative or embedded task.
                if (not target or re.search(r"这个|那个|之前|上面|还是|或者|比较|对比|" + SKU_PATTERN, target, re.I)):
                    pending.append(clause)
                    continue
                if target not in self.conditions:
                    self.conditions = (*self.conditions, target)
                    added.append(target)
                self.primary_condition = target
                explicit_edit = True
                focused = target
                self._source("condition:" + target, clause)
                self._source("conditions", clause)
        self.pending_conditions = tuple(pending) if pending else (() if explicit_edit else self.pending_conditions)
        self.condition_changes = {
            "added": tuple(c for c in added if c in self.conditions and c not in before),
            "retained": tuple(c for c in before if c in self.conditions),
            "removed": tuple(c for c in removed if c not in self.conditions),
            "pending": self.pending_conditions,
            "focused": (focused,) if focused in self.conditions else (),
        }
        if self.pending_conditions and not self.query_field:
            self.clarification = "请明确要新增、修改或撤销的具体场景条件；尚未确认的条件不参与比较。"

    def update(self, text: str, *, restricted: bool = False) -> None:
        """Apply user context only; restricted input must be P0-1's safe excerpt.

        restricted mode preserves SKU/reference context but does not promote any
        goal or recommendation preference from a medically rejected request.
        """
        self.clarification = None
        self.name_query = None
        self.condition_changes = {}
        old_field = self.query_field
        new_topic = bool(re.search(r"换个话题|新话题|重新开始|聊点别的|(?:另|重新)(?:买|选)|换个需求", text))
        # A new named shopping need without a SKU/reference starts a fresh task.
        # General off-topic questions are recognized only by these explicit forms.
        reference_pattern = r"前者|后者|另一个|它|这款|那款|这个(?!条件|场景)(?:商品)?"
        if not re.search(SKU_PATTERN, text, re.I) and not re.search(reference_pattern, text):
            new_topic |= bool(re.search(r"什么是|是什么|帮我(?:写|翻译|计算)|写(?:一段|个).*(?:代码|程序)", text))
            if self.current_sku or self.comparison_skus or self.goal:
                new_topic |= "想买" in text
        if new_topic:
            self._clear_product()
            self.budget_max = None
            self.goal = None
            self.excluded_directions = ()
            self.conditions = ()
            self.primary_condition = None
            self.pending_conditions = ()
            self.sources.clear()
            old_field = None

        ids = list(dict.fromkeys(s.upper() for s in re.findall(SKU_PATTERN, text, re.I)))
        # Remove explicitly rejected identities before deciding whether there is
        # a comparison. In particular, correction is not a two-product selection.
        rejected = {
            s.upper() for s in re.findall(
                rf"(?:不是|不要(?:再)?(?:看|推荐)?|不看|排除)\s*({SKU_PATTERN})", text, re.I,
            )
        }
        positive_ids = [s for s in ids if s not in rejected]
        compare = any(term in text for term in ("比较", "对比", "区别", "差别", "哪个", "怎么选", "如何选"))
        reverse_order = bool(re.search(r"顺序(?:反过来|反转|倒过来)|反过来比较|交换(?:一下)?(?:比较)?顺序", text))
        if not ids and reverse_order:
            self.current_sku = None
            self.sources.pop("current_sku", None)
            if len(self.comparison_skus) == 2:
                self.comparison_skus = self.comparison_skus[::-1]
                self._source("comparison_skus", text)
            else:
                self.clarification = "尚未确定比较的两个商品，请提供 SKU 和比较顺序。"
        elif len(positive_ids) == 2 and (compare or (
            len(self.comparison_skus) == 2
            and re.search(r"我说错了|更正|改为|改成|换成", text)
        )):
            self.comparison_skus = tuple(positive_ids)
            self.current_sku = None
            self.sources.pop("current_sku", None)
            self._source("comparison_skus", text)
        elif len(positive_ids) == 1:
            self.current_sku = positive_ids[0]
            if self.current_sku not in self.comparison_skus:
                self.comparison_skus = ()
                self.sources.pop("comparison_skus", None)
            self._source("current_sku", text, "safe_input_reference" if restricted else "user_explicit")
        elif len(positive_ids) > 1:
            self._clear_product()
            self.clarification = "你提到了多个商品，请明确要查看哪个 SKU，或说明比较顺序。"
        elif rejected:
            if self.current_sku in rejected:
                self.current_sku = None
                self.sources.pop("current_sku", None)
            if rejected.intersection(self.comparison_skus):
                self.comparison_skus = ()
                self.sources.pop("comparison_skus", None)

        if not ids and not re.search(reference_pattern, text) and re.search(r"改看|换成|切换到|我说的是", text):
            self._clear_product()
            self.name_query = _extract_name_query(text)
            self.clarification = "请提供新商品的 SKU；当前未能可靠关联商品身份。"

        reference = re.search(reference_pattern, text)
        if not ids and reference:
            word = reference.group()
            resolved = None
            references = set(re.findall(reference_pattern, text))
            if len(references) > 1:
                self.clarification = "请明确要查看哪个商品，或说明是否要继续比较。"
            elif word in ("前者", "后者") and len(self.comparison_skus) == 2:
                resolved = self.comparison_skus[0 if word == "前者" else 1]
            elif word == "另一个" and len(self.comparison_skus) == 2 and self.current_sku in self.comparison_skus:
                resolved = next(s for s in self.comparison_skus if s != self.current_sku)
            elif word not in ("前者", "后者", "另一个"):
                resolved = self.current_sku
            self.current_sku = resolved
            if resolved:
                self._source("current_sku", text, "comparison_reference" if word in ("前者", "后者", "另一个") else "current_product_reference")
            else:
                self.sources.pop("current_sku", None)
                self.clarification = "你指的是哪个商品？请提供 SKU。"

        fields = [name for name, terms in FIELD_TERMS.items() if any(t in text for t in terms)]
        if not ids and fields and "delivery" not in fields:
            # A short bare field question can use the selection. An extra noun
            # (e.g. an unresolved new product name) must not silently use an old SKU.
            remainder = re.sub(reference_pattern, "", text)
            for terms in FIELD_TERMS.values():
                for term in terms:
                    remainder = remainder.replace(term, "")
            remainder = re.sub(r"再问一次|再说一遍|再说一次|我问的是|只想知道|只比较|只比|有点贵|太贵|超出我的预算|超出预算|超预算|请问|麻烦|帮我|告诉我|查询|看一下|只看|直接说|多少|如何|是|为|的|呢|吗|呀|啊|[\s，,。！？!?；;]", "", remainder)
            if remainder:
                self._clear_product()
                self.name_query = _extract_name_query(text)
                self.clarification = "请提供要查询商品的 SKU；当前未能可靠关联商品身份。"
        if (not ids and not fields and not reference and not restricted
                and not re.search(r"有点贵|太贵|预算|再想想|考虑|不用|谢谢|知道|明白", text)
                and re.fullmatch(r"[^，,。！？!?；;]+呢[？?]?", text.strip())):
            # A new name followed by '呢' must not inherit the old selection.
            self._clear_product()
            self.name_query = _extract_name_query(text)
            self.clarification = "请确认新商品的完整名称或 SKU。"
        self.query_field = fields[0] if len(fields) == 1 else None
        if not fields and len(positive_ids) == 1 and re.search(r"呢[？?]?$", text.strip()):
            self.query_field = old_field
        if self.query_field:
            self._source("query_field", text, "user_explicit" if fields else "previous_field_followup")
        else:
            self.sources.pop("query_field", None)

        if not restricted:
            self._update_conditions(text)
            for clause in re.split(r"[，,。！？!?；;]+", text):
                if re.search(r"预算(?:不限|不限制)|(?:取消|撤销|去掉)预算(?:限制)?|不限预算", clause):
                    self.budget_max = None
                    self._source("budget_max", clause, "user_retraction")
                else:
                    budget = re.search(r"预算\s*(?:改成|改为|为|是|在|只有|不超过|最多)?\s*(\d+(?:\.\d+)?)\s*元?", clause)
                    if not budget and self.budget_max is not None:
                        budget = re.fullmatch(r"\s*改(?:成|为)\s*(\d+(?:\.\d+)?)\s*元(?:以内|以下)?\s*", clause)
                    if budget:
                        self.budget_max = float(budget[1])
                        self._source("budget_max", clause)
            goal = re.search(r"(?:目标(?:需求)?(?:是|为)|需求(?:是|改成)|主要(?:是|想要|用于)|想买)\s*([^，,。！？!?；;]+)", text)
            if goal:
                self.goal = goal[1].strip()
                self._source("goal", goal[0])
            if re.search(r"(?:取消|撤销|去掉)(?:目标|需求)", text):
                self.goal = None
                self._source("goal", text, "user_retraction")
            exclusion = re.search(r"不要(?:再)?推荐\s*([^，,。！？!?；;]+)", text)
            if exclusion:
                self.excluded_directions = tuple(dict.fromkeys((*self.excluded_directions, exclusion[1].strip())))
                self._source("excluded_directions", exclusion[0])
            if re.search(r"(?:取消|撤销|去掉)(?:推荐)?排除(?:限制)?|取消不推荐限制", text):
                self.excluded_directions = ()
                self._source("excluded_directions", text, "user_retraction")

        # A specific field query with no product after reset/new-topic is not a
        # license to revive the product from older chat messages. Platform delivery
        # questions remain handled by the existing Agent's tool policy.
        if self.query_field in ("price", "spec", "suitable_crowd", "taboo", "stock"):
            if self.current_sku is None and not (
                self.comparison_skus and (compare or len(self.comparison_skus) == 2)
            ):
                self.clarification = "请确认要查询哪个商品，并提供 SKU。"

    def resolve_name(self, query: str, sku: str | None, clarification: str | None) -> None:
        """Program-only resolution of this turn's pending name, not an LLM setter."""
        if self.name_query != query:
            raise ValueError("Name resolution does not match the pending query")
        self.current_sku = sku
        self.comparison_skus = ()
        self.sources.pop("comparison_skus", None)
        self.sources.pop("current_sku", None)
        self.clarification = clarification
        if sku:
            self._source("current_sku", query, "unique_name_search")

    def model_context(self) -> dict[str, Any]:
        """Return a plain snapshot without mutable aliases or product facts."""
        return asdict(self)
