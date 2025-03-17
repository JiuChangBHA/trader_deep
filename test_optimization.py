"""
Simple test to verify optimization changes
"""
import asyncio
import logging
from core.algo_trading_system import Backtester, config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_test():
    logger.info("Creating backtester with optimized settings...")
    backtester = Backtester(config)
    backtester.initialize()
    
    logger.info("Running backtest with optimized settings...")
    await backtester.run()
    
    logger.info("Backtest completed successfully!")
    return backtester

if __name__ == "__main__":
    logger.info("Starting optimization test")
    backtester = asyncio.run(run_test())
    logger.info("Test completed") 