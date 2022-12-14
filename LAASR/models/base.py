from typing import Optional, AnyStr, Any
import torch
import torchaudio
import warnings

class AudioModelBase:
    """
    Class base to use audio models.
    """
    def __init__(
        self,
        model_path: Optional[str] = '',
        sampling_rate: Optional[int] = 16000,
        chunk_length: Optional[int] = 5000,
        device_str: Optional[str] = 'auto',
        torchscript: Optional[bool] = False,
        FP16: Optional[bool] = False,
        *args,
        **kwargs,
        ) -> None:
        self.model_path = model_path
        self.sampling_rate = sampling_rate
        self.chunk_length = chunk_length
        self.torchscript = torchscript
        self.device = self.get_device(device_str)
        self.args = args
        self.FP16 = FP16
        self.kwargs = kwargs
        self.model = None

    def __select_backend_device__(self):
        if torch.cuda.is_available():
            self.backend = 'cuda'
            return torch.device('cuda')
        elif torch.has_mps:
            self.backend = 'mps'
            return torch.device('mps')
        self.backend = 'cpu'
        return torch.device('cpu')

    def get_device(self, device_str: str):
        if not device_str:
            raise UserWarning("Device is not compatible")
        if device_str == 'auto':
            return self.__select_backend_device__()
        return torch.device(device_str)

    def __load_torchscript__(self, checkpoint_str: str):
        self.model = torch.jit.load(checkpoint_str)

    def __load_model_pytorch__(self, checkpoint_str: str, strict=True):
        if self.torchscript:
            self.__load_torchscript__(checkpoint_str)
        else:
            checkpoint = torch.load(checkpoint_str, map_location=self.device)
            self.model.load_state_dict(checkpoint, strict=strict)
        self.model.eval()
        self.model.to(self.device)

    def read_audio(self, path: str):
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sampling_rate:
            transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
            wav = transform(wav)
            sr = self.sampling_rate
        assert sr == self.sampling_rate
        return wav.squeeze(0)

    def split_audio(self, audio: torch.Tensor):
        pass

    def load_model(self):
        raise NotImplementedError("Load Model not implemented yet.")

    def validate_tensor(self, input):
        if not torch.is_tensor(input):
            try:
                input = torch.Tensor(input)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")
        return input

    def validate_audio_mono(self, audio):
        if len(audio.shape) > 1:
            for i in range(len(audio.shape)):  # trying to squeeze empty dimensions
                audio = audio.squeeze(0)
            if len(audio.shape) > 1:
                raise ValueError("More than one dimension in audio. Are you trying to process audio with 2 channels?")

    def validate_sample_rate(self):
        sampling_rate = self.sampling_rate
        if sampling_rate > self.sampling_rate and (sampling_rate % self.sampling_rate == 0):
            step = sampling_rate // self.sampling_rate
            sampling_rate = self.sampling_rate
            audio = audio[::step]
            warnings.warn(f'Sampling rate is a multiply of {self.sampling_rate}, casting to {self.sampling_rate} manually!')
        else:
            step = 1
        return step

    def preprocess_input(self, *args):
        new_args = []
        for arg in args:
            if torch.is_tensor(arg):
                new_arg = arg.to(self.device)
                if self.FP16:
                    new_arg = new_arg.half()
                new_args.append(new_arg)
            else:
                new_args.append(arg)
        return new_args

    def forward(self):
        raise NotImplementedError("Forward not implemented yet.")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)