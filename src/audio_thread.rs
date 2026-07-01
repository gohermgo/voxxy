use crate::ResamplerCfg;
use heapless::spsc::Producer;

pub struct AudioThread {
    cpal_stream: cpal::Stream,
}

impl AudioThread {
    pub fn new(
        dev: cpal::Device,
        mut sample_tx: Producer<'static, f32>,
    ) -> anyhow::Result<(AudioThread, ResamplerCfg)> {
        use cpal::traits::DeviceTrait;

        let stream_cfg = dev.default_input_config()?;

        assert_eq!(
            stream_cfg.sample_rate(),
            cpal::SAMPLE_RATE_CD,
            "input stream is not CD sample-rate"
        );

        tracing::info!(
            "input device gave us {}hz, {}ch",
            stream_cfg.sample_rate(),
            stream_cfg.channels()
        );

        let resampler_cfg = ResamplerCfg {
            input_rate: stream_cfg.sample_rate(),
            // HIFI mode, LOFI is 11025 member THAT
            target_rate: 16_000,
            // arbitrary, tune for latency/throughput later
            chunk_size: 1024,
        };

        // thread is managed here by cpal for us tbh
        let stream = dev.build_input_stream(
            stream_cfg.config(),
            move |values: &[f32], _callback_info| {
                // we want to deinterleave, only care about half the values
                for val in values.iter().step_by(2) {
                    let Ok(()) = sample_tx.enqueue(*val) else {
                        tracing::error!("failed to enqueue...");
                        break;
                    };
                }
            },
            |e| {
                tracing::error!("in the error callback!! {e}");
            },
            None,
        )?;

        Ok((
            AudioThread {
                cpal_stream: stream,
            },
            resampler_cfg,
        ))
    }

    pub fn play(&self) -> Result<(), cpal::Error> {
        use cpal::traits::StreamTrait;
        self.cpal_stream.play()
    }
}
