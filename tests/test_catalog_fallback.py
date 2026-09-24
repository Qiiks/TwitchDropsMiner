from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.services.inventory_service as inventory_service_module
from src.config import State
from src.services.inventory_service import InventoryService


# `src.services.catalog` is the module this change adds, so it is imported
# lazily: a module-level import would break collection on the pre-change tree
# and hide the regression test below behind a collection error.
try:
    from src.services.catalog import PublicCatalog
except ImportError:  # pragma: no cover - only on the pre-change tree
    PublicCatalog = None  # type: ignore[assignment, misc]

requires_catalog = pytest.mark.skipif(
    PublicCatalog is None, reason="src.services.catalog does not exist yet"
)


class _FakeResponse:
    def __init__(self, *, status: int = 200, payload: object = None, error: Exception | None = None):
        self.status = status
        self._payload = payload
        self._error = error

    async def json(self) -> object:
        if self._error is not None:
            raise self._error
        return self._payload


class _ResponseContext:
    def __init__(self, response: _FakeResponse):
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        return None


class _FakeSession:
    closed = False

    def __init__(self, response: _FakeResponse):
        self._response = response

    def get(self, url: str) -> _ResponseContext:
        return _ResponseContext(self._response)

    async def close(self) -> None:
        self.closed = True


def _catalog(response: _FakeResponse) -> PublicCatalog:
    catalog = PublicCatalog("https://catalog.invalid")
    catalog._session = _FakeSession(response)
    return catalog

_CATALOG_SHAPE = {
    "id": "c1",
    "name": "From Catalog",
    "status": "ACTIVE",
    "accountLinkURL": "https://example.invalid/link",
    "startAt": "2026-09-20T00:00:00Z",
    "endAt": "2026-10-20T00:00:00Z",
    "game": {"id": "1", "displayName": "G", "name": "G", "slug": "g"},
    "timeBasedDrops": [],
    # what PublicCatalog normalises an absent channel list to
    "allow": {"channels": None, "isEnabled": True},
    "self": {"isAccountConnected": True},
}

# Same campaign as Twitch reports it: a participating-channel list, so the two
# sources disagree on the type of allow.channels.
_TWITCH_SHAPE = {
    **_CATALOG_SHAPE,
    "name": "From Twitch",
    "allow": {"channels": [{"id": "9", "name": "chan", "displayName": "chan"}], "isEnabled": True},
    "self": {"isAccountConnected": False},
}


def _record(campaign_id: str, **overrides: object) -> dict[str, object]:
    """Build a mirror record that carries every key PublicCatalog validates."""
    return {**_CATALOG_SHAPE, "id": campaign_id, **overrides}



@requires_catalog
@pytest.mark.asyncio
async def test_public_catalog_flattens_campaign_groups_in_order():
    catalog = _catalog(
        _FakeResponse(
            payload=[
                {"gameDisplayName": "First Game", "rewards": [_record("first"), _record("second")]},
                {"gameDisplayName": "Second Game", "rewards": [_record("third")]},
            ]
        )
    )

    campaigns = await catalog.campaigns()

    assert [campaign["id"] for campaign in campaigns] == ["first", "second", "third"]


@requires_catalog
@pytest.mark.asyncio
async def test_public_catalog_defaults_self_without_overwriting_existing_self():
    existing_self = {"isAccountConnected": False, "extra": "preserved"}
    catalog = _catalog(
        _FakeResponse(
            payload=[
                {
                    "rewards": [
                        _record("missing-self"),
                        _record("existing-self", self=existing_self),
                    ]
                }
            ]
        )
    )

    campaigns = await catalog.campaigns()

    assert campaigns[0]["self"] == {"isAccountConnected": True}
    assert campaigns[1]["self"] == existing_self


@requires_catalog
@pytest.mark.asyncio
async def test_public_catalog_normalizes_allow_defaults_and_preserves_channels():
    catalog = _catalog(
        _FakeResponse(
            payload=[
                {
                    "rewards": [
                        _record("missing-allow"),
                        _record("missing-channels", allow={"isEnabled": False}),
                        _record(
                            "existing-channels",
                            allow={"channels": ["channel-a"], "isEnabled": False},
                        ),
                    ]
                }
            ]
        )
    )

    campaigns = await catalog.campaigns()

    # `None` (not `[]`) matches what Twitch sends for a campaign without a
    # participating-channel list; an empty list collides with it in
    # GQLClient.merge_data and aborts the inventory fetch.
    assert campaigns[0]["allow"] == {"channels": None, "isEnabled": True}
    assert campaigns[1]["allow"] == {"channels": None, "isEnabled": False}
    assert campaigns[2]["allow"] == {"channels": ["channel-a"], "isEnabled": False}


@requires_catalog
@pytest.mark.asyncio
async def test_public_catalog_handles_invalid_payloads_and_skips_malformed_groups():
    invalid_responses = (
        _FakeResponse(status=503, payload=[]),
        _FakeResponse(error=ValueError("invalid JSON")),
        _FakeResponse(payload={"rewards": []}),
    )
    for response in invalid_responses:
        assert await _catalog(response).campaigns() == []

    campaigns = await _catalog(
        _FakeResponse(
            payload=[
                "not a group",
                {"rewards": "not a list"},
                {"rewards": ["not a reward", _record("valid")]},
            ]
        )
    ).campaigns()

    assert [campaign["id"] for campaign in campaigns] == ["valid"]


@pytest.mark.asyncio
async def test_fetch_campaigns_skips_gated_campaign_and_returns_resolved_data(monkeypatch):
    monkeypatch.delenv("TDM_CATALOG_URL", raising=False)
    twitch = SimpleNamespace(
        get_auth=AsyncMock(return_value=SimpleNamespace(user_id="123")),
        gql_request=AsyncMock(
            return_value=[
                {"data": {"user": {"dropCampaign": None}}},
                {
                    "data": {
                        "user": {
                            "dropCampaign": {
                                "id": "resolved",
                                "details": "from-twitch",
                            }
                        }
                    }
                },
            ]
        ),
    )
    service = InventoryService(twitch)

    campaigns = await service.fetch_campaigns(
        [
            ("gated", {"id": "gated", "name": "overview-gated"}),
            ("resolved", {"id": "resolved", "name": "overview-resolved"}),
        ]
    )

    assert service._catalog is None
    assert campaigns["resolved"]["details"] == "from-twitch"


@requires_catalog
@pytest.mark.asyncio
async def test_inventory_service_uses_catalog_for_missing_campaigns_and_empty_inventory(
    monkeypatch,
):
    twitch_campaign = {"id": "twitch-id", "name": "Twitch list"}
    catalog_campaign = {"id": "catalog-id", "name": "Catalog detail", "catalog": True}
    duplicate_catalog_campaign = {
        "id": "twitch-id",
        "name": "Catalog duplicate",
        "catalog": True,
    }
    twitch = SimpleNamespace(
        get_auth=AsyncMock(return_value=SimpleNamespace(user_id="123")),
        gql_request=AsyncMock(
            return_value=[
                {
                    "data": {
                        "user": {
                            "dropCampaign": {
                                "id": "twitch-id",
                                "name": "Twitch detail",
                                "detail": "from-twitch",
                            }
                        }
                    }
                }
            ]
        ),
    )
    service = object.__new__(InventoryService)
    service._twitch = twitch
    service._catalog = SimpleNamespace(
        campaigns=AsyncMock(return_value=[twitch_campaign, catalog_campaign, duplicate_catalog_campaign])
    )

    campaigns = await service.fetch_campaigns(
        [
            ("twitch-id", twitch_campaign),
            ("catalog-id", {"id": "catalog-id", "name": "Twitch overview"}),
        ]
    )

    assert campaigns["twitch-id"]["name"] == "Twitch list"
    assert campaigns["twitch-id"]["detail"] == "from-twitch"
    assert "catalog" not in campaigns["twitch-id"]
    assert campaigns["catalog-id"]["catalog"] is True

    class _FakeCampaign:
        def __init__(self, twitch_client, data, claimed_benefits):
            self.id = data["id"]
            self.drops = []
            self.active = False
            self.upcoming = False
            self.starts_at = 0
            self.ends_at = 0
            self.eligible = False
            self.time_triggers = []

        def can_earn_within(self, when):
            return False

    monkeypatch.setattr(inventory_service_module, "DropsCampaign", _FakeCampaign)
    catalog_inventory_campaign = {
        "id": "catalog-inventory-id",
        "status": "ACTIVE",
        "game": {"id": "game-id"},
    }
    inventory_twitch = SimpleNamespace(
        gql_request=AsyncMock(
            side_effect=[
                {
                    "data": {
                        "currentUser": {
                            "inventory": {
                                "dropCampaignsInProgress": [],
                                "gameEventDrops": [],
                            }
                        }
                    }
                },
                {"data": {"currentUser": {"dropCampaigns": []}}},
            ]
        ),
        gui=SimpleNamespace(
            status=SimpleNamespace(update=MagicMock()),
            inv=SimpleNamespace(clear=MagicMock(), add_campaign=AsyncMock()),
        ),
        _drops={},
        _campaigns={},
        inventory=[],
        _mnt_triggers=[],
        _mnt_task=None,
        _state=State.IDLE,
        _maintenance_service=SimpleNamespace(run_maintenance_task=AsyncMock()),
    )
    inventory_service = object.__new__(InventoryService)
    inventory_service._twitch = inventory_twitch
    inventory_service._catalog = SimpleNamespace(
        campaigns=AsyncMock(return_value=[catalog_inventory_campaign])
    )
    inventory_service.fetch_campaigns = AsyncMock(
        return_value={catalog_inventory_campaign["id"]: catalog_inventory_campaign}
    )

    await inventory_service.fetch_inventory()
    await inventory_twitch._mnt_task

    inventory_service._catalog.campaigns.assert_awaited_once_with()
    inventory_service.fetch_campaigns.assert_awaited_once_with(
        [(catalog_inventory_campaign["id"], catalog_inventory_campaign)]
    )



@requires_catalog
def test_merge_campaign_data_survives_mixed_source_shape_clash():
    """A fallback list and a Twitch detail can disagree in shape without crashing."""
    only_catalog = {**_CATALOG_SHAPE, "id": "c2", "name": "Catalog Only"}

    merged = InventoryService._merge_campaign_data(
        {"c1": dict(_CATALOG_SHAPE), "c2": only_catalog},
        {"c1": dict(_TWITCH_SHAPE)},
    )

    assert sorted(merged) == ["c1", "c2"]
    # Twitch answered for c1, so its record wins outright.
    assert merged["c1"]["name"] == "From Twitch"
    # c2 was never answered for, so the catalog record is kept as-is.
    assert merged["c2"]["name"] == "Catalog Only"


@requires_catalog
@pytest.mark.asyncio
async def test_twitch_account_state_overrides_catalog_linkage_placeholder():
    """The mirror cannot know linkage, so it must never beat Twitch's real answer."""
    twitch = SimpleNamespace(
        get_auth=AsyncMock(return_value=SimpleNamespace(user_id="123")),
        gql_request=AsyncMock(
            return_value=[{"data": {"user": {"dropCampaign": dict(_TWITCH_SHAPE)}}}]
        ),
    )
    service = object.__new__(InventoryService)
    service._twitch = twitch
    service._catalog = None

    result = await service.fetch_campaigns([("c1", dict(_CATALOG_SHAPE))])

    assert result["c1"]["self"]["isAccountConnected"] is False


@requires_catalog
@pytest.mark.asyncio
async def test_public_catalog_drops_records_the_campaign_model_cannot_index():
    """Truncated records are dropped at the boundary rather than raising later."""
    valid = {**_CATALOG_SHAPE, "id": "good-2"}
    catalog = _catalog(
        _FakeResponse(
            payload=[
                {
                    "rewards": [
                        {"name": "no id", "status": "ACTIVE"},
                        {"id": "x", "name": "no status"},
                        {"id": "", "name": "empty id", "status": "ACTIVE"},
                        {**_CATALOG_SHAPE, "id": "y", "timeBasedDrops": None},
                        valid,
                    ]
                }
            ]
        )
    )

    campaigns = await catalog.campaigns()

    assert [campaign["id"] for campaign in campaigns] == ["good-2"]


@requires_catalog
@pytest.mark.asyncio
async def test_available_campaign_filter_skips_records_without_id_or_status():
    """A malformed campaign list entry must not abort the whole inventory fetch."""
    inventory = {
        "dropCampaignsInProgress": [],
        "gameEventDrops": [],
    }
    twitch = SimpleNamespace(
        gql_request=AsyncMock(
            side_effect=[
                {"data": {"currentUser": {"inventory": inventory}}},
                {
                    "data": {
                        "currentUser": {
                            "dropCampaigns": [
                                {"name": "no id", "status": "ACTIVE"},
                                {"id": "keep", "status": "ACTIVE", "game": {"id": "g"}},
                                {"id": "expired", "status": "EXPIRED"},
                            ]
                        }
                    }
                },
            ]
        ),
        gui=SimpleNamespace(
            status=SimpleNamespace(update=MagicMock()),
            inv=SimpleNamespace(clear=MagicMock(), add_campaign=AsyncMock()),
        ),
        _drops={},
        _campaigns={},
        inventory=[],
        _mnt_triggers=[],
        _mnt_task=None,
        _state=State.IDLE,
        _maintenance_service=SimpleNamespace(run_maintenance_task=AsyncMock()),
    )
    service = object.__new__(InventoryService)
    service._twitch = twitch
    service._catalog = None
    service.fetch_campaigns = AsyncMock(return_value={})

    await service.fetch_inventory()
    await twitch._mnt_task

    service.fetch_campaigns.assert_awaited_once_with([("keep", {"id": "keep", "status": "ACTIVE", "game": {"id": "g"}})])
