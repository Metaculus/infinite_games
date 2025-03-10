# -- DO NOT TOUCH BELOW - ENV SET --
# flake8: noqa: E402
import asyncio
import os
import sys
import typing

# Force torch - must be set before importing bittensor
os.environ["USE_TORCH"] = "1"

# Add the parent directory of the script to PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
# -- DO NOT TOUCH ABOVE --

import time

from bittensor import logging
from forecasting_tools import BinaryQuestion, MainBot, QuestionState

from neurons.miner.forecasters.base import BaseForecaster, DummyForecaster
from neurons.miner.forecasters.llm_forecaster import LLMForecaster
from neurons.miner.main import Miner
from neurons.miner.models.event import MinerEvent
from neurons.validator.utils.logger.logger import InfiniteGamesLogger, miner_logger


class MetaculusMiner(BaseForecaster):
    def __init__(self, event: MinerEvent, logger: InfiniteGamesLogger, extremize: bool = False):
        super().__init__(event, logger, extremize)
        self.bot = MainBot(
            research_reports_per_question=1,
            predictions_per_research_report=5,
        )

    async def _run(self) -> float | int:
        question = BinaryQuestion(
            question_text=self.event.get_description(),
            background_info=None,
            resolution_criteria=None,
            fine_print=None,
            id_of_post=0,
            state=QuestionState.OPEN,
        )
        try:
            reports = await self.bot.forecast_questions([question])
            probability = reports[0].prediction
        except Exception as e:
            self.logger.error(f"Error forecasting question with llm: {e}")
            probability = 0.5
        return probability


def get_forecaster(logger: InfiniteGamesLogger):
    async def assign_forecaster(event: MinerEvent) -> typing.Type[BaseForecaster]:
        return MetaculusMiner(
            event,
            logger=logger,
            extremize=False,
        )

    return assign_forecaster


if __name__ == "__main__":
    start_time = time.time()

    async def run_miner() -> None:
        miner_logger.start_session()

        miner = Miner(logger=miner_logger, assign_forecaster=get_forecaster(miner_logger))
        await miner.initialize()
        with miner as miner:
            while True:
                miner_logger.debug(f"Miner running for {time.time() - start_time:.1f} seconds")
                await asyncio.sleep(5)

    asyncio.run(run_miner())
