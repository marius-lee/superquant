#!/usr/bin/env python3
"""日线增量同步 — Sina/TickFlow → market.db。来源: quant/data/store.py DataStore.update_daily()"""
import os, sys
sys.path.insert(0, os.path.expanduser("~/project/quant"))
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

from data.store import DataStore

if __name__ == '__main__':
    ds = DataStore()
    total = ds.update_daily()
    print(f"日线同步完成: {total} 新行")
    ds.close()
