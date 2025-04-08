from pathlib import Path
from typing import Iterable

from neurons.validator.db.client import DatabaseClient
from neurons.validator.models.event import EVENTS_FIELDS, EventsModel, EventStatus
from neurons.validator.models.miner import MINERS_FIELDS, MinersModel
from neurons.validator.models.prediction import (
    PREDICTION_FIELDS,
    PredictionExportedStatus,
    PredictionsModel,
)
from neurons.validator.models.score import SCORE_FIELDS, ScoresModel
from neurons.validator.utils.logger.logger import InfiniteGamesLogger

SQL_FOLDER = Path(Path(__file__).parent, "sql")


class DatabaseOperations:
    __db_client: DatabaseClient
    logger: InfiniteGamesLogger

    def __init__(self, db_client: DatabaseClient, logger: InfiniteGamesLogger):
        if not isinstance(db_client, DatabaseClient):
            raise ValueError("Invalid db_client arg")

        if not isinstance(logger, InfiniteGamesLogger):
            raise TypeError("logger must be an instance of InfiniteGamesLogger.")

        self.__db_client = db_client
        self.logger = logger

    async def delete_event(self, event_id: str) -> Iterable[tuple[str]]:
        return await self.__db_client.delete(
            """
                DELETE FROM events WHERE event_id = ? RETURNING event_id
            """,
            [event_id],
        )

    async def delete_predictions(self, batch_size: int) -> Iterable[tuple[int]]:
        return await self.__db_client.delete(
            """
                WITH predictions_to_delete AS (
                    SELECT
                        p.ROWID
                    FROM
                        predictions p
                    LEFT JOIN
                        events e ON p.unique_event_id = e.unique_event_id
                    WHERE
                        (
                            e.unique_event_id IS NULL
                            OR (
                                    e.processed = TRUE
                                    AND datetime(e.resolved_at) < datetime(CURRENT_TIMESTAMP, '-4 day')
                                )
                            OR e.status = ?
                        )
                        AND p.exported = ?
                    ORDER BY
                        p.ROWID ASC
                    LIMIT ?
                )
                DELETE FROM
                    predictions
                WHERE
                    ROWID IN (
                        SELECT
                            ROWID
                        FROM
                            predictions_to_delete
                    )
                RETURNING
                    ROWID
            """,
            [EventStatus.DISCARDED, PredictionExportedStatus.EXPORTED, batch_size],
        )

    async def get_event(self, unique_event_id: str) -> None | EventsModel:
        result = await self.__db_client.one(
            f"""
                SELECT
                    {', '.join(EVENTS_FIELDS)}
                FROM events
                WHERE
                    unique_event_id = ?
            """,
            parameters=[unique_event_id],
            use_row_factory=True,
        )

        if result is None:
            return None

        return EventsModel(**dict(result))

    async def get_events_last_resolved_at(self) -> str | None:
        row = await self.__db_client.one(
            """
                SELECT MAX(resolved_at) FROM events
            """
        )

        if row is not None:
            return row[0]

    async def get_events_pending_first_created_at(self) -> str | None:
        row = await self.__db_client.one(
            """
                SELECT MIN(created_at) FROM events WHERE status = ?
            """,
            [EventStatus.PENDING],
        )

        if row is not None:
            return row[0]

    async def get_events_to_predict(self) -> Iterable[tuple[str]]:
        return await self.__db_client.many(
            """
                SELECT
                    event_id,
                    market_type,
                    description,
                    cutoff,
                    resolve_date,
                    end_date,
                    metadata
                FROM
                    events
                WHERE
                    status = ?
                    AND datetime(CURRENT_TIMESTAMP) < datetime(cutoff)
            """,
            parameters=[EventStatus.PENDING],
        )

    async def get_last_event_from(self) -> str | None:
        row = await self.__db_client.one(
            """
                SELECT MAX(created_at) FROM events
            """
        )

        if row is not None:
            return row[0]

    async def get_miners_count(self) -> int:
        row = await self.__db_client.one(
            """
                SELECT COUNT(*) FROM miners
            """
        )

        return row[0]

    async def get_predictions_to_export(self, current_interval_minutes: int, batch_size: int):
        return await self.__db_client.many(
            """
                SELECT
                    p.ROWID,
                    p.unique_event_id,
                    p.minerHotkey,
                    p.minerUid,
                    e.event_type,
                    p.predictedOutcome,
                    p.interval_start_minutes,
                    p.interval_agg_prediction,
                    p.interval_count,
                    p.submitted
                FROM
                    predictions p
                JOIN
                    events e ON e.unique_event_id = p.unique_event_id
                WHERE
                    p.exported = ?
                    AND p.interval_start_minutes < ?
                ORDER BY
                    p.ROWID ASC
                LIMIT
                    ?
            """,
            [PredictionExportedStatus.NOT_EXPORTED, current_interval_minutes, batch_size],
        )

    async def mark_predictions_as_exported(self, ids: list[str]):
        placeholders = ", ".join(["?"] * len(ids))

        return await self.__db_client.update(
            f"""
                UPDATE
                    predictions
                SET
                    exported = ?
                WHERE
                    ROWID IN ({placeholders})
                RETURNING
                    ROWID
            """,
            [PredictionExportedStatus.EXPORTED] + ids,
        )

    async def resolve_event(
        self, event_id: str, outcome: str, resolved_at: str
    ) -> Iterable[tuple[str]]:
        return await self.__db_client.update(
            """
                UPDATE
                    events
                SET
                    status = ?,
                    outcome = ?,
                    resolved_at = ?,
                    local_updated_at = CURRENT_TIMESTAMP
                WHERE
                    event_id = ?
                    AND status = ?
                RETURNING
                    event_id
            """,
            [EventStatus.SETTLED, outcome, resolved_at, event_id, EventStatus.PENDING],
        )

    async def upsert_events(self, events: list[list[any]]) -> None:
        return await self.__db_client.insert_many(
            """
                INSERT INTO events
                    (
                        unique_event_id,
                        event_id,
                        market_type,
                        event_type,
                        description,
                        starts,
                        resolve_date,
                        outcome,
                        status,
                        metadata,
                        created_at,
                        cutoff,
                        end_date,
                        registered_date,
                        local_updated_at
                    )
                VALUES
                    (
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        CURRENT_TIMESTAMP,
                        CURRENT_TIMESTAMP
                    )
                ON CONFLICT
                    (unique_event_id)
                DO NOTHING
            """,
            events,
        )

    async def upsert_miners(self, miners: list[list[any]]) -> None:
        return await self.__db_client.insert_many(
            """
                INSERT INTO miners
                    (
                        miner_uid,
                        miner_hotkey,
                        node_ip,
                        registered_date,
                        last_updated,
                        blocktime,
                        blocklisted,
                        is_validating,
                        validator_permit
                    )
                VALUES
                    (
                        ?,
                        ?,
                        ?,
                        ?,
                        CURRENT_TIMESTAMP,
                        ?,
                        FALSE,
                        ?,
                        ?
                    )
                ON CONFLICT
                    (miner_hotkey, miner_uid)
                DO UPDATE
                    set node_ip = ?,
                    last_updated = CURRENT_TIMESTAMP,
                    blocktime = ?,
                    is_validating = excluded.is_validating,
                    validator_permit = excluded.validator_permit
            """,
            miners,
        )

    async def upsert_predictions(self, predictions: list[list[any]]):
        return await self.__db_client.insert_many(
            """
                INSERT INTO predictions (
                    unique_event_id,
                    minerHotkey,
                    minerUid,
                    predictedOutcome,
                    interval_start_minutes,
                    interval_agg_prediction,
                    blocktime,
                    interval_count,
                    submitted
                )
                VALUES (
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    1,
                    CURRENT_TIMESTAMP
                )
                ON CONFLICT(unique_event_id,  interval_start_minutes, minerUid)
                DO UPDATE SET
                    interval_agg_prediction = (interval_agg_prediction * interval_count + ?) / (interval_count + 1),
                    interval_count = interval_count + 1
            """,
            predictions,
        )

    async def upsert_pydantic_events(self, events: list[EventsModel]) -> None:
        """Same as upsert_events but with pydantic models"""

        fields_to_insert = [
            field_name
            for field_name in EVENTS_FIELDS
            if field_name not in ("registered_date", "local_updated_at")
        ]
        placeholders = ", ".join(["?"] * len(fields_to_insert))
        columns = ", ".join(fields_to_insert)

        # Convert each event into a tuple of values in the same order as fields_to_insert
        event_tuples = [
            tuple(getattr(event, field_name) for field_name in fields_to_insert) for event in events
        ]

        sql = f"""
                INSERT INTO events
                    ({columns}, registered_date, local_updated_at)
                VALUES
                    ({placeholders}, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT
                    (unique_event_id)
                DO NOTHING
        """
        return await self.__db_client.insert_many(
            sql=sql,
            parameters=event_tuples,
        )

    async def get_events_for_scoring(self, max_events=1000) -> list[EventsModel]:
        """
        Returns all events that were recently resolved and need to be scored
        """

        rows = await self.__db_client.many(
            f"""
                SELECT
                    {', '.join(EVENTS_FIELDS)}
                FROM events
                WHERE status = ?
                    AND outcome IS NOT NULL
                    AND processed = false
                ORDER BY resolved_at ASC
                LIMIT ?
            """,
            parameters=[EventStatus.SETTLED, max_events],
            use_row_factory=True,
        )

        events = []
        for row in rows:
            try:
                event = EventsModel(**dict(row))
                events.append(event)
            except Exception:
                self.logger.exception("Error parsing event", extra={"row": row})

        return events

    async def get_predictions_for_event(
        self, unique_event_id: str, interval_start_minutes: int
    ) -> list[PredictionsModel]:
        rows = await self.__db_client.many(
            f"""
                SELECT
                    {', '.join(PREDICTION_FIELDS)}
                FROM
                    predictions
                WHERE
                    unique_event_id = ?
                    AND interval_start_minutes = ?
                ORDER BY
                    CAST(minerUid AS INTEGER) ASC,
                    minerHotkey ASC
            """,
            parameters=[unique_event_id, interval_start_minutes],
            use_row_factory=True,
        )

        predictions = []

        for row in rows:
            try:
                prediction = PredictionsModel(**dict(row))
                predictions.append(prediction)
            except Exception:
                self.logger.exception("Error parsing prediction", extra={"row": row})

        return predictions

    async def get_predictions_for_scoring(self, unique_event_id: str) -> list[PredictionsModel]:
        rows = await self.__db_client.many(
            f"""
                SELECT
                    {', '.join(PREDICTION_FIELDS)}
                FROM predictions
                WHERE unique_event_id = ?
            """,
            parameters=(unique_event_id,),
            use_row_factory=True,
        )

        predictions = []
        for row in rows:
            try:
                prediction = PredictionsModel(**dict(row))
                predictions.append(prediction)
            except Exception:
                self.logger.exception("Error parsing prediction", extra={"row": row})

        return predictions

    async def get_miners_last_registration(self) -> list:
        rows = await self.__db_client.many(
            f"""
                WITH ranked AS (
                    SELECT
                        {', '.join(MINERS_FIELDS)},
                        ROW_NUMBER() OVER (
                            PARTITION BY miner_uid
                            ORDER BY registered_date DESC
                        ) AS rn
                    FROM miners t
                )
                SELECT
                    {', '.join(MINERS_FIELDS)}
                FROM ranked
                WHERE rn = 1
                ORDER BY miner_uid
            """,
            use_row_factory=True,
        )
        miners = []
        for row in rows:
            try:
                miner = MinersModel(**dict(row))
                miners.append(miner)
            except Exception:
                self.logger.exception("Error parsing miner", extra={"row": row})

        return miners

    async def mark_event_as_processed(self, unique_event_id: str) -> None:
        return await self.__db_client.update(
            """
                UPDATE events
                SET processed = true
                WHERE unique_event_id = ?
            """,
            parameters=(unique_event_id,),
        )

    async def mark_event_as_exported(self, unique_event_id: str) -> None:
        return await self.__db_client.update(
            """
                UPDATE events
                SET exported = true
                WHERE unique_event_id = ?
            """,
            parameters=(unique_event_id,),
        )

    async def mark_event_as_discarded(self, unique_event_id: str) -> None:
        """For resolved events which cannot be scored"""
        return await self.__db_client.update(
            """
                UPDATE events
                SET status = ?
                WHERE unique_event_id = ?
            """,
            parameters=[EventStatus.DISCARDED, unique_event_id],
        )

    async def insert_peer_scores(self, scores: list[ScoresModel]) -> None:
        """Insert raw peer scores into the scores table"""

        fields_to_insert = [
            "event_id",
            "miner_uid",
            "miner_hotkey",
            "prediction",
            "event_score",
            "spec_version",
        ]
        placeholders = ", ".join("?" for _ in fields_to_insert)
        columns = ", ".join(fields_to_insert)

        # Convert each event into a tuple of values in the same order as fields_to_insert
        score_tuples = [
            tuple(getattr(score, field_name) for field_name in fields_to_insert) for score in scores
        ]

        sql = f"""
                INSERT INTO scores ({columns})
                VALUES ({placeholders})
                ON CONFLICT
                    (event_id, miner_uid, miner_hotkey)
                DO UPDATE SET
                    prediction = excluded.prediction,
                    event_score = excluded.event_score,
                    spec_version = excluded.spec_version
        """
        return await self.__db_client.insert_many(
            sql=sql,
            parameters=score_tuples,
        )

    async def get_events_for_metagraph_scoring(self, max_events: int = 1000) -> list[dict]:
        """
        Returns all events that were recently peer scored and not processed.
        These events need to be ordered by row_id for the moving average calculation.
        """

        rows = await self.__db_client.many(
            """
                SELECT
                    event_id,
                    MIN(ROWID) AS min_row_id
                FROM scores
                WHERE processed = false
                GROUP BY event_id
                ORDER BY min_row_id ASC
                LIMIT ?
            """,
            use_row_factory=True,
            parameters=[
                max_events,
            ],
        )

        events = []
        for row in rows:
            try:
                event = dict(row)
                events.append(event)
            except Exception:
                self.logger.exception("Error parsing event", extra={"row": row})
        return events

    async def set_metagraph_peer_scores(self, event_id: str, n_events: int) -> list:
        """
        Calculate the moving average of peer scores for a given event
        """
        raw_sql = Path(SQL_FOLDER, "metagraph_peer_score.sql").read_text()
        updated = await self.__db_client.update(
            raw_sql,
            parameters={"event_id": event_id, "n_events": n_events},
        )

        return updated

    async def get_peer_scored_events_for_export(self, max_events: int = 1000) -> list[EventsModel]:
        """
        Get peer scored events that have not been exported
        """
        ev_event_fields = ["ev." + field for field in EVENTS_FIELDS]
        rows = await self.__db_client.many(
            f"""
                WITH events_to_export AS (
                    SELECT
                        event_id,
                        MIN(ROWID) AS min_row_id
                    FROM scores
                    WHERE processed = 1
                        AND exported = 0
                    GROUP BY event_id
                    ORDER BY min_row_id ASC
                    LIMIT ?
                )
                SELECT
                    {', '.join(ev_event_fields)}
                FROM events ev
                JOIN events_to_export ete ON ev.event_id = ete.event_id
                ORDER BY ete.min_row_id ASC
            """,
            use_row_factory=True,
            parameters=[
                max_events,
            ],
        )

        events = []
        for row in rows:
            try:
                event = EventsModel(**dict(row))
                events.append(event)
            except Exception:
                self.logger.exception("Error parsing event", extra={"row": row})

        return events

    async def get_peer_scores_for_export(self, event_id: str) -> list:
        """
        Get peer scores for a given event
        Processed has to be true, to guarantee that metagraph score is set
        """
        rows = await self.__db_client.many(
            f"""
                SELECT
                    {', '.join(SCORE_FIELDS)}
                FROM scores
                WHERE event_id = ?
                    AND processed = 1
            """,
            parameters=[event_id],
            use_row_factory=True,
        )

        scores = []
        for row in rows:
            try:
                score = ScoresModel(**dict(row))
                scores.append(score)
            except Exception:
                self.logger.exception("Error parsing score", extra={"row": row})

        return scores

    async def mark_peer_scores_as_exported(self, event_id: str) -> list:
        """
        Mark peer scores from event_id as exported
        """
        return await self.__db_client.update(
            """
                UPDATE scores
                SET exported = 1
                WHERE event_id = ?
            """,
            parameters=(event_id,),
        )

    async def get_last_metagraph_scores(self) -> list:
        """
        Returns the last known metagraph_score for each miner_uid, miner_hotkey;
        We cannot simply take from the last event - could be an old event scored now, so
        if the miner registered after the event cutoff, we will have no metagraph_score
        """
        rows = await self.__db_client.many(
            f"""
                WITH grouped AS (
                    SELECT miner_uid AS g_miner_uid,
                        miner_hotkey AS g_miner_hotkey,
                        MAX(ROWID) AS max_rowid
                    FROM scores
                    WHERE processed = 1
                        AND created_at > datetime(CURRENT_TIMESTAMP, '-10 day')
                    GROUP BY miner_uid, miner_hotkey
                )
                SELECT
                    {', '.join(SCORE_FIELDS)}
                FROM scores s
                JOIN grouped
                    ON s.miner_uid = grouped.g_miner_uid
                    AND s.miner_hotkey = grouped.g_miner_hotkey
                    AND s.ROWID = grouped.max_rowid
            """,
            use_row_factory=True,
        )

        scores = []
        for row in rows:
            try:
                score = ScoresModel(**dict(row))
                scores.append(score)
            except Exception:
                self.logger.exception("Error parsing score", extra={"row": row})

        return scores

    async def vacuum_database(self, pages: int):
        await self.__db_client.script(f"PRAGMA incremental_vacuum({pages})")

    async def get_wa_prediction_event(
        self, unique_event_id: str, interval_start_minutes: int
    ) -> float | None:
        """
        Retrieve the weighted average of the latest predictions for a given event
        """
        raw_sql = Path(SQL_FOLDER, "latest_predictions_event.sql").read_text()
        row = await self.__db_client.one(
            raw_sql,
            parameters={
                "unique_event_id": unique_event_id,
                "interval_start_minutes": interval_start_minutes,
            },
        )

        if row[0] is None:
            self.logger.warning(
                "No predictions found for the event",
                extra={
                    "unique_event_id": unique_event_id,
                    "interval_start_minutes": interval_start_minutes,
                },
            )

            return row[0]

        weighted_average_prediction = float(row[0])

        return weighted_average_prediction
