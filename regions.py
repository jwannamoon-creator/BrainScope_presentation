REGIONS = {
    # Emotion circuit
    "vmPFC_OFC": {
        "pos": (-18, 42, -8),
        "desc": "복내측 전전두엽/안와전두피질: 감정 조절, 보상 평가, 편도체 조절"
    },
    "ACC": {
        "pos": (-5, 28, 24),
        "desc": "전대상피질: 감정 조절, 갈등 감지, 자기 관련 처리"
    },
    "Amygdala": {
        "pos": (-22, -5, -18),
        "desc": "편도체: 공포, 불안, 정서적 중요성 처리"
    },
    "Insula": {
        "pos": (-38, 0, 5),
        "desc": "섬엽: 내부 감각, 정서 인식, 신체 상태 통합"
    },
    "Hippocampus": {
        "pos": (-28, -25, -12),
        "desc": "해마: 기억 형성, 맥락 처리, 감정 기억"
    },

    # Memory circuit
    "dlPFC": {
        "pos": (-35, 38, 28),
        "desc": "배외측 전전두엽: 작업기억, 인지 조절, 실행 기능"
    },
    "Parahippocampal": {
        "pos": (-24, -35, -14),
        "desc": "해마방회: 맥락 기억, 장소 정보, 해마 주변 기억 처리"
    },
    "PCC": {
        "pos": (-5, -45, 28),
        "desc": "후대상피질: 자전적 기억, 회상, default mode network"
    },
    "Angular": {
        "pos": (-42, -60, 35),
        "desc": "각회: 의미 기억, 기억 회상, 언어·의미 처리"
    },
}


CIRCUITS = {
    "Emotion Circuit": {
        "regions": [
            "vmPFC_OFC",
            "ACC",
            "Amygdala",
            "Insula",
            "Hippocampus"
        ],
        "edges": [
            ("vmPFC_OFC", "ACC"),
            ("vmPFC_OFC", "Amygdala"),
            ("ACC", "Amygdala"),
            ("Amygdala", "Insula"),
            ("Amygdala", "Hippocampus"),
            ("Hippocampus", "vmPFC_OFC")
        ]
    },

    "Memory Circuit": {
        "regions": [
            "dlPFC",
            "Hippocampus",
            "Parahippocampal",
            "PCC",
            "Angular"
        ],
        "edges": [
            ("dlPFC", "Hippocampus"),
            ("Hippocampus", "Parahippocampal"),
            ("Hippocampus", "PCC"),
            ("PCC", "Angular"),
            ("Angular", "dlPFC")
        ]
    }
}