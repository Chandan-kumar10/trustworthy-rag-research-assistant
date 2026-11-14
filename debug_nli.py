import numpy as np
from sentence_transformers import CrossEncoder

m = CrossEncoder("cross-encoder/nli-deberta-v3-base")
prem = ("Experimental results demonstrate up to a 30% improvement over baseline models, "
        "highlighting the framework's effectiveness, scalability, and suitability.")
for h in ["CARROT achieves up to a 30% improvement over baseline models.",
          "CARROT achieves up to a 63% improvement over baseline models."]:
    s = np.array(m.predict([(prem, h)]))
    e = np.exp(s - s.max()); p = e / e.sum()
    print(h); print("  probs:", np.round(p, 3), " id2label:", getattr(m.model.config, "id2label", None))