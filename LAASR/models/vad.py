import torch
from .base import AudioModelBase
from typing import Optional, AnyStr, Any, List
import warnings
from tqdm import tqdm
try:
    from torch.cuda.amp import autocast
except:
    pass
import os

class VAD(AudioModelBase):
    """
    Simplified VAD class based on SileroVAD:

    https://github.com/snakers4/silero-vad

    """
    def __init__(
        self,
        model_path: Optional[str] = './LAASR/checkpoints/silero_vad.jit',
        sampling_rate: Optional[int] = 16000, 
        chunk_length: Optional[int] = 5000,
        torch_hub_repo: Optional[str] = 'snakers4/silero-vad',
        torch_hub_model: Optional[str] = 'silero_vad',
        FP16: Optional[bool] = False,
        threshold: Optional[float] = 0.5,
        min_speech_duration_ms: Optional[int] = 50,
        min_silence_duration_ms: Optional[int] = 100,
        window_size_samples: Optional[int] = 512,
        speech_pad_ms: Optional[int] = 30,
        return_seconds: Optional[bool] = False,
        torchscript: Optional[bool] = True,
        *args, 
        **kwargs
        ) -> None:
        super().__init__(model_path=model_path, sampling_rate=sampling_rate, chunk_length=chunk_length, FP16=FP16, torchscript=torchscript, *args, **kwargs)
        self.torch_hub_repo = torch_hub_repo
        self.torch_hub_model = torch_hub_model
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_samples = window_size_samples
        self.speech_pad_ms = speech_pad_ms
        self.return_seconds = return_seconds
        self.load_model()

    def init_jit_model(self, model_path: str):
        model = torch.jit.load(model_path, map_location=self.device)
        return model

    def load_model(self):
        if self.model_path != '':
            self.model = self.init_jit_model(self.model_path)
        else:
            self.model, _ = torch.hub.load(repo_or_dir=self.torch_hub_repo,
                                                    model=self.torch_hub_model,
                                                    force_reload=True)
        self.model.eval()
        self.model.to(self.device)
        if self.FP16:
            self.model = self.model.half()

    def from_file(self, path:str, **kwargs: Any) -> Any:
        self.filename = os.path.basename(path)
        audio = self.read_audio(path)
        return self(audio, **kwargs)

    def validate_window_size(self):
        if self.sampling_rate == 8000 and self.self.window_size_samples > 768:
            warnings.warn('self.window_size_samples is too big for 8000 sampling_rate! Better set self.window_size_samples to 256, 512 or 768 for 8000 sample rate!')
        if self.self.window_size_samples not in [256, 512, 768, 1024, 1536]:
            warnings.warn('Unusual self.window_size_samples! Supported self.window_size_samples:\n - [512, 1024, 1536] for 16000 sampling_rate\n - [256, 512, 768] for 8000 sampling_rate')

    def postprocess_output(self, speech_probs, audio_length_samples, return_type='samples'):
        """
        This method is used for splitting long audios into speech chunks using silero VAD
        Parameters
        ----------
        audio: torch.Tensor, one dimensional
            One dimensional float torch.Tensor, other types are casted to torch if possible
        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 sample rates
        min_speech_duration_ms: int (default - 250 milliseconds)
            Final speech chunks shorter min_speech_duration_ms are thrown out
        max_speech_duration_s: int (default -  inf)
            Maximum duration of speech chunks in seconds
            Chunks longer than max_speech_duration_s will be split at the timestamp of the last silence that lasts more than 100s (if any), to prevent agressive cutting.
            Otherwise, they will be split aggressively just before max_speech_duration_s.
        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it
        window_size_samples: int (default - 1536 samples)
            Audio chunks of window_size_samples size are fed to the silero VAD model.
            WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000 sample rate and 256, 512, 768 samples for 8000 sample rate.
            Values other than these may affect model perfomance!!
        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)
        Returns
        ----------
        speeches: list of dicts
            list containing ends and beginnings of speech chunks (samples or seconds based on return_seconds)
        """


        """
        Postprocess speech VAD probabilities 

        Args:
            speech_probs (_type_): _description_
            audio_length_samples (_type_): _description_
            return_type (str, optional): .Defaults to 'samples', Posibles ['samples', 'seconds', 'ms'].

        Returns:
            _type_: _description_
        """        
        min_speech_samples = self.sampling_rate * (self.min_speech_duration_ms / 1000)
        speech_pad_samples = self.sampling_rate * self.speech_pad_ms / 1000
        min_silence_samples = self.sampling_rate * self.min_silence_duration_ms / 1000
        min_silence_samples_at_max_speech = self.sampling_rate * 98 / 1000
        triggered = False
        speeches = []
        current_speech = {}
        neg_threshold = self.threshold - 0.15
        temp_end = 0 # to save potential segment end (and tolerate some silence)
        prev_end = next_start = 0 # to save potential segment limits in case of maximum segment size reached

        for i, speech_prob in enumerate(speech_probs):
            if (speech_prob >= self.threshold) and temp_end:
                temp_end = 0
                if next_start < prev_end:
                    next_start = self.window_size_samples * i

            if (speech_prob >= self.threshold) and not triggered:
                triggered = True
                current_speech['start'] = self.window_size_samples * i
                continue

            if (speech_prob < neg_threshold) and triggered:
                if not temp_end:
                    temp_end = self.window_size_samples * i
                if ((self.window_size_samples * i) - temp_end) > min_silence_samples_at_max_speech : # condition to avoid cutting in very short silence
                    prev_end = temp_end
                if (self.window_size_samples * i) - temp_end < min_silence_samples:
                    continue
                else:
                    current_speech['end'] = temp_end
                    if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                        speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    continue

        if current_speech and (audio_length_samples - current_speech['start']) > min_speech_samples:
            current_speech['end'] = audio_length_samples
            speeches.append(current_speech)

        for i, speech in enumerate(speeches):
            if i == 0:
                speech['start'] = int(max(0, speech['start'] - speech_pad_samples))
            if i != len(speeches) - 1:
                silence_duration = speeches[i+1]['start'] - speech['end']
                if silence_duration < 2 * speech_pad_samples:
                    speech['end'] += int(silence_duration // 2)
                    speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - silence_duration // 2))
                else:
                    speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))
                    speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - speech_pad_samples))
            else:
                speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))
        if return_type.lower() == 'samples':
            return speeches
        convert = 1 if return_type.lower() =='seconds' else 1000
        for speech_dict in speeches:
            speech_dict['start'] = (speech_dict['start'] / self.sampling_rate) *convert
            speech_dict['end'] = (speech_dict['end'] / self.sampling_rate) * convert
        return speeches


    def forward(self, audio: torch.Tensor) -> List[float]:    
        audio = self.validate_tensor(audio)
        self.validate_audio_mono(audio)
        self.model.reset_states()
        audio_length_samples = len(audio)
        speech_probs = []
        for current_start_sample in tqdm(range(0, audio_length_samples, self.window_size_samples), desc=f"Processing: {self.filename}"):
            chunk = audio[current_start_sample: current_start_sample + self.window_size_samples]
            if len(chunk) < self.window_size_samples:
                chunk = torch.nn.functional.pad(chunk, (0, int(self.window_size_samples - len(chunk))))
            chunk = chunk.to(self.device)
            if self.FP16 and self.backend == 'cuda':
                chunk = chunk.half()
                with torch.no_grad(), autocast():
                    speech_prob = self.model(chunk, self.sampling_rate).item()
            else:
                with torch.no_grad():
                    speech_prob = self.model(chunk, self.sampling_rate).item()
            speech_probs.append(speech_prob)
        return speech_probs

    def __call__(self, audio: torch.Tensor, **kwargs: Any) -> Any:
        probs = self.forward(audio)
        return self.postprocess_output(probs, audio.shape[-1], **kwargs)

