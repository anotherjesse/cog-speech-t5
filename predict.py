import os
from typing import List
import soundfile as sf
import time

import torch
import numpy as np
from cog import BasePredictor, Input, Path
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset

import settings

MODEL_ID = "stabilityai/stable-diffusion-2-1"
MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"


hf_embeddings = {
    "BDL": "spkemb/cmu_us_bdl_arctic-wav-arctic_a0009.npy",
    "CLB": "spkemb/cmu_us_clb_arctic-wav-arctic_a0144.npy",
    "KSP": "spkemb/cmu_us_ksp_arctic-wav-arctic_b0087.npy",
    "RMS": "spkemb/cmu_us_rms_arctic-wav-arctic_b0353.npy",
    "SLT": "spkemb/cmu_us_slt_arctic-wav-arctic_a0508.npy",
}


class Predictor(BasePredictor):
    def setup(self):
        self.processor = SpeechT5Processor.from_pretrained(
            settings.MODEL_NAME, cache_dir=settings.MODEL_CACHE
        )
        self.model = SpeechT5ForTextToSpeech.from_pretrained(
            settings.MODEL_NAME, cache_dir=settings.MODEL_CACHE
        )
        # self.embeddings_dataset = load_dataset(
        #     "Matthijs/cmu-arctic-xvectors", split="validation", cache_dir=settings.MODEL_CACHE
        # )
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            settings.VOCODER_NAME, cache_dir=settings.MODEL_CACHE
        )


    @torch.inference_mode()
    def predict(
        self,
        text: str = Input(
            description="Input text to speak",
            default="hello world",
        ),
        voice: str = Input(
            description="Speaker to use",
            default="BDL",
            choices=list(hf_embeddings.keys()))
    ) -> Path:
        """Run a single prediction on the model"""
        if os.path.exists("output.wav"):
            os.remove("output.wav")

        start_time = time.time()
        speaker_embeddings = torch.tensor(np.load(hf_embeddings[voice])).unsqueeze(0)

        # speaker_embeddings = torch.tensor(
        #     self.embeddings_dataset[embedding]["xvector"]
        # ).unsqueeze(0)
        print("speaker embeddings time", time.time() - start_time)
        
        start_time = time.time()
        inputs = self.processor(text=text, return_tensors="pt")
        print("processor time", time.time() - start_time)

        start_time = time.time()
        speech = self.model.generate_speech(
            inputs["input_ids"], speaker_embeddings, vocoder=self.vocoder
        )
        print("generate speech time", time.time() - start_time)

        sf.write("output.wav", speech.numpy(), samplerate=16000)

        return Path("output.wav")
