from __future__ import annotations

from typing import Any

from refund_abuse_risk.model.two_head import (
    device_cluster_score_from_features,
    entity_prior_from_features,
    entity_scores_from_features,
    link_scores_from_features,
)
from refund_abuse_risk.schemas.models import EntityScores, LinkScores


def derive_entity_link_scores(
    features: dict[str, Any],
) -> tuple[EntityScores, LinkScores, float, float]:
    entity = EntityScores(**entity_scores_from_features(features))
    links = LinkScores(**link_scores_from_features(features))
    device = device_cluster_score_from_features(features)
    prior = entity_prior_from_features(features)
    return entity, links, device, prior
