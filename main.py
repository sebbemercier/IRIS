# Copyright 2026 The OpenSLM Project
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

import sentencepiece as spm

from IRIS.config import settings

class IrisAnalytics:
    def __init__(self, tokenizer_path=None):
        t_path = tokenizer_path or settings.TOKENIZER_PATH
        self.sp = spm.SentencePieceProcessor(model_file=t_path)

    def analyze_stock_velocity(self, sales_data, stock_count):
        print("IRIS: Analyse de la vitesse de vente...")
        if stock_count < 10:
            return "ALERTE: Réapprovisionnement nécessaire immédiatement."
        return "Statut: Stock sain."

if __name__ == "__main__":
    iris = IrisAnalytics()
    print(iris.analyze_stock_velocity([], 5))