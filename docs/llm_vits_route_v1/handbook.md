# LLM × VITS: Component Replacement and Verifiable Theory

**Full parallel bilingual edition · Research and implementation specification · Literature checked through 2026-10-27**

Scope: six component proposals S1–S6, five theory proposals T1–T5, shared data and engineering protocols, step-by-step derivations, tensor dimensions, training/inference procedures, falsifiable experiments, and submission-readiness assessment. Neither language edition is an abridgment; proposal, equation, and reference identifiers match.

**Status.** This is an implementation-oriented research design, not a completed paper or a pretrained TTS product. Proposed values are initial settings, not validated optima. No MOS, WER, memory, or training-throughput results are fabricated. The companion code validates small modules and numerical theory experiments locally; it does not constitute reproduction of six complete speech systems. Conference acceptance cannot be guaranteed; each proposal states the evidence still needed for a competitive submission.

## Reading and execution route

Complete Chapters 1–3 on notation, data, and the baseline before selecting one component proposal. The recommended streaming route is S6 + T2 + T3: stateful stochastic duration, acoustically relevant error bounds, and trajectory-level calibrated commitment. For generative modeling, choose S3 + T4: continuous AudioVAE and bounded-memory causal flows. S1, S2, and S4 are separate projects; S5 is usually an auxiliary experiment. Combining all eleven directions is not a substitute for a clear contribution.

## Corrections to the earlier discussion

VITS2 is not identical to the original flow-based VITS SDP: its duration branch uses adversarial training, so the original variational dequantization likelihood cannot be relabeled as its objective. The original official VITS is the tractable-density baseline here; VITS2 is a separate strong comparator [R01–R03]. StyleTTS 2 is an advanced TTS architecture, not automatically a billion-parameter foundation model; its WavLM backbone is a directly transferable pretrained speech representation [R04,R15].

Unknown future must be defined relative to the information actually available. With a frozen upstream model, full KV state, and greedy decoding, future text may be computationally uncomputed rather than probabilistically random. Raw softmax probabilities need not describe a temperature- or top-p-transformed deployment process. The principal setting uses an explicit stochastic decoding kernel or an empirical external-text process. Speculative text is for planning only; unobserved words must never be spoken.

Low-rate AudioVAE/codec representations require a redesigned alignment clock when incompatible with phoneme durations. CFM velocity regression is not a VAE KL term. A GAN generator alone does not supply an evaluable waveform likelihood. The SDP likelihood bound cannot be inserted into an allegedly exact mutual information estimator. This handbook separates normalized probabilistic models, computable approximations, and engineering objectives throughout.

# 1. Shared notation, tensor dimensions, and a rigorous baseline

## 1.1 Clocks and shape conventions

Derivations use a single example; code adds batch dimension B. Audio tensors are channels-first; Transformer tensors are batch/sequence/channel. All masks participate in losses and lengths. The new experimental baseline uses 24 kHz, hop 256, and FFT 1024; these are design choices, not the original VITS defaults. Do not silently change the sampling rate of a pretrained waveform decoder.

| Symbol | Shape / proposed value | Meaning of axes |
|---|---|---|
| $B,N,L,T,S$ | Positive scalar integers | Batch, phonemes, text tokens, acoustic frames, waveform samples |
| $y$ | $\mathbb R^{B\times1\times S}$ | Real mono waveform |
| $x,m$ | $B\times513\times T$; $B\times80\times T$ | Linear magnitude spectrum; log Mel |
| $c,h$ | $B\times N$; $B\times192\times N$ | Phoneme IDs; phoneme hidden states |
| $H^{LM}$ | $B\times L\times2048$ | Qwen3-1.7B states; 2048 is verified in its config [R16] |
| $A,d$ | $B\times N\times T$; $B\times1\times N$ | Monotonic alignment; integer frame durations |
| $z,z_p$ | $B\times192\times T$ | Acoustic-posterior space; flow base space |
| $r,\rho$ | $B\times64$; $B\times16$ | Block future-prosody latent; rhythm state |
| $W,H_k$ | Variable-length tokens; structured observable state | Future suffix; information available at decision k, not a fixed-size tensor |
| $Y$ | $\mathbb R^q$, $q=6n_*$ | Measurable prosody of a fixed pending phone block; proposed $n_*=8$ |
| $\theta,\phi,\psi,\alpha,\beta$ | Parameter collections, no single shape | Prior, posterior, prosody, duration, and state parameters; risk level is separately $a_{risk}$ |

Below, $\mu$ and $\sigma$ match the associated variable shape; positive $\sigma$ is a standard deviation, not a variance. $I_d$ is the $d\times d$ identity; $\odot$ is elementwise multiplication. KL, expected losses, and log densities are scalar. A $\lambda$ is a nonnegative scalar weight. $\operatorname{vec}$ flattens nonbatch axes. $\tau$ indexes diffusion/ODE time; $k$ indexes streaming decisions.

$$
A_{it}\in\{0,1\},\quad \sum_iA_{it}=1,\quad d_i=\sum_tA_{it},\quad T=\sum_i d_i,\quad h^{frm}=hA.
\tag{G1}
$$

**Symbols, dimensions, and conditions.** $A\in\{0,1\}^{N\times T}$, $h\in\mathbb R^{192\times N}$, $h^{frm}\in\mathbb R^{192\times T}$. Nonpadding phones require $d_i\ge1$ and a monotonic non-skipping path, hence $T\ge N$.

## 1.2 LLM conditioning and prefix availability

$$
\bar H=P_{ph\leftarrow tok}H^{LM},\qquad \tilde h^{\mathsf T}=h^{\mathsf T}+\sigma_g(U_g[ h^{\mathsf T},\bar H])\odot(\bar H U_L).
\tag{G2}
$$

**Symbols, dimensions, and conditions.** $P_{ph\leftarrow tok}\in\mathbb R^{N\times L}$ is row-normalized using character spans and G2P alignment; $\bar H\in\mathbb R^{N\times2048}$ and $U_L\in\mathbb R^{2048\times192}$. $U_g$ denotes an affine map producing $N\times192$ values and $\sigma_g$ is sigmoid. Process only stable, normalized phonemes; do not leak future punctuation, full-sentence G2P, or future-token states.

Freeze the text LLM and cache prefix states; train a new 2048-to-192 adapter. Record actual temperature, top-p, EOS, and suffix horizon. Independently sampled deployment-distribution candidates receive equal weight 1/M; normalized beam candidates are not unbiased Monte Carlo samples. Start with offline M=8 and a 16-text-token horizon; use online M=2 or distill into a lightweight mixture head. Include prediction computation in end-to-end latency. Llama-VITS and LLM2Speech are direct precedents [R33,R34]; this interface alone is not novel.

## 1.3 Conditional VAE: joint density to ELBO

To avoid treating a GAN waveform function as an explicit density, define a Mel observation model first. Use MAS-derived durations $d^*$ as fixed pseudo-observations. The derivation bounds the joint $p(m,d^*|H)$, not an exact alignment-marginalized waveform likelihood. Duration determines dimension; each $d$ has its density on the corresponding $T=\sum_i d_i$ subspace.

$$
p(m,d,r,z\mid H)=p_\psi(r\mid H)p_\alpha(d\mid H,r)p_\theta(z\mid H,r,d)p_\gamma(m\mid z,H),
\tag{G3}
$$

**Symbols, dimensions, and conditions.** $r\in\mathbb R^{64}$, $d\in\mathbb N_+^N$, $z\in\mathbb R^{192\times T}$, and $m\in\mathbb R^{80\times T}$. Every factor must be a normalized density or mass function.

$$
\begin{aligned}q(r,z)&=q_\eta(r\mid y,c_{full})q_\phi(z\mid x,r,d^*),\\
\log p(m,d^*\mid H)&=\log\int q(r,z)\frac{p(m,d^*,r,z\mid H)}{q(r,z)}\,dr\,dz\\
&\ge\mathbb E_q[\log p(m,d^*,r,z\mid H)-\log q(r,z)].\end{aligned}
\tag{G4}
$$

**Symbols, dimensions, and conditions.** $q_\eta$ is the new prosody posterior and $q_\phi$ the acoustic posterior. Require compatible supports and finite integrals. The third line follows from concavity of log/Jensen. A training teacher may see complete text, whereas the prior sees only $H$.

$$
\begin{aligned}\mathcal E=&\mathbb E_q\log p_\gamma(m\mid z,H)+\mathbb E_{q_\eta}\log p_\alpha(d^*\mid H,r)\\
&-\mathrm{KL}(q_\eta(r)\Vert p_\psi(r\mid H))\\
&-\mathbb E_{q_\eta}\mathrm{KL}(q_\phi(z\mid x,r,d^*)\Vert p_\theta(z\mid H,r,d^*)).\end{aligned}
\tag{G5}
$$

**Symbols, dimensions, and conditions.** $\mathcal E$ is a scalar per-example ELBO. Substitute the four log factors of G3 into G4, then collect the two $\mathbb E(\log q-\log p)$ expressions as KL terms. The entropy of $q_\eta$ cannot be omitted.

$$
\begin{aligned}\widehat m&=\operatorname{Mel}(G_\gamma(z)),\\
p_\gamma(m\mid z)&=\prod_{a=1}^{80}\prod_{t=1}^{T}\frac1{2b}\exp\left(-\frac{|m_{at}-\widehat m_{at}|}{b}\right),\\
-\log p_\gamma(m\mid z)&=\|m-\widehat m\|_1/b+80T\log(2b).\end{aligned}
\tag{G6}
$$

**Symbols, dimensions, and conditions.** $G_\gamma$ outputs a waveform; $\widehat m$ matches $m$. The fixed scalar $b>0$ is a Laplace scale. Sum valid frames. This is a Mel-space density, not a waveform density. Frame averaging and arbitrary GAN/feature-matching weights change the objective from this exact ELBO.

## 1.4 Normalizing flow, Gaussian KL, and SDP

$$
\begin{aligned}z_p&=f_\theta(z;H,r,d),\\
\log p_\theta(z\mid H,r,d)&=\log\mathcal N(\operatorname{vec}z_p;\mu_p,\operatorname{diag}\sigma_p^2)+\log|\det J_f|,\\
\mathrm{KL}(q_\phi\Vert p_\theta)&=\mathbb E_{z\sim q_\phi}[\log q_\phi(z)-\log p_\theta(z)].\end{aligned}
\tag{G7}
$$

**Symbols, dimensions, and conditions.** $\mu_p,\sigma_p\in\mathbb R^{192T}$. Never explicitly construct $J_f\in\mathbb R^{192T\times192T}$; sum coupling-block log determinants. Reparameterize $z=\mu_q+\sigma_q\odot\epsilon$, $\epsilon\sim N(0,I_{192T})$. Original acoustic mean-only coupling has zero logdet; general affine coupling does not [R02].

$$
\mathrm{KL}(N(\mu_q,\operatorname{diag}s_q^2)\Vert N(\mu_p,\operatorname{diag}s_p^2))=\frac12\sum_{j=1}^{D}\left[2\log\frac{s_{p,j}}{s_{q,j}}+\frac{s_{q,j}^2+(\mu_{q,j}-\mu_{p,j})^2}{s_{p,j}^2}-1\right].
\tag{G8}
$$

**Symbols, dimensions, and conditions.** Both distributions occupy the same D-dimensional space; means/scales are D-vectors. Expand Gaussian log densities using $E_q[(X_j-\mu_{p,j})^2]=s_{q,j}^2+(\mu_{q,j}-\mu_{p,j})^2$. This closed form is not valid for arbitrary mixtures or an untransformed flow prior.

$$
\begin{aligned}p_\alpha(d\mid c)&=\int_{(0,1)^N}\int_{\mathbb R^N}p_\alpha(d-u,v\mid c)\,dv\,du,\\
\log p_\alpha(d\mid c)&\ge\mathbb E_{q_\omega(u,v\mid d,c)}[\log p_\alpha(d-u,v\mid c)-\log q_\omega(u,v\mid d,c)].\end{aligned}
\tag{G9}
$$

**Symbols, dimensions, and conditions.** Here c denotes duration conditioning, including H and r. $d,u,v\in\mathbb R^N$, with integer-positive d, $u\in(0,1)^N$, and auxiliary v. Apply $\log a$ after $a=d-u>0$ and retain its Jacobian. The first line integrates a continuous cell to obtain discrete mass; insert q/q and apply Jensen for the second.

The original SDP returns a negative duration bound, not an exact log probability for arbitrary d. Substituting G9 into the positive log-duration term of G5 yields a looser valid bound. Theory experiments requiring exact discrete probabilities should add a categorical duration head or discretized-logistic mass model; this is a new estimator, not a capability already provided by SDP. Original code detaches text and speaker conditions; inject the new prosody branch after detachment or supervise it separately so duration gradients can reach it [R02].

The engineering objective may be written $-\mathcal E+\lambda_{adv}L_{adv}+\lambda_{fm}L_{fm}+\lambda_{aux}L_{aux}$. All L and lambda values are scalar; VITS waveform discrimination and feature matching can remain. Report the probabilistic subobjective, practical optimization objective, and estimators separately rather than labeling all of them an exact ELBO.

# 2. Shared data, features, and leakage prevention

## 2.1 Three data budgets

| Tier | Data | Purpose and limitation |
|---|---|---|
| P0 | LJSpeech 1.1: 13,100 clips, about 24 hours, one speaker [R27] | Interface/overfitting checks, not evidence of multilingual or unseen-speaker generalization |
| P1 | Official LibriTTS-R train/dev/test partitions [R28] | Main study; preserve official speaker separation; compute actual hours rather than interpreting train-clean-100 literally |
| P1-ZH | AISHELL-3: about 85 hours, 218 speakers [R29] | Independent Mandarin validation with consistent tones, erhua, and polyphone handling |
| P2 | An authorized fixed Emilia subset, e.g. 500 hours [R30] | Scale only after P1 gains; track subset/version/license; Emilia and Emilia-YODAS licenses differ |
| Evaluation | Seed-TTS-Eval EN/ZH plus a new prefix benchmark [R31] | Evaluation only; exclude test text and reference speakers from teacher-synthetic training |

License summaries do not replace per-file review: LJSpeech is described as public-domain in the US; LibriTTS-R uses CC BY 4.0; the AISHELL-3 resource page lists Apache 2.0. Save download dates, licenses, provenance, and filtering manifests. Use appropriately licensed or consented voices; synthetic teacher data is not automatically unrestricted. Prevent leakage through neighboring test sentences, same-chapter excerpts, and reused reference recordings.

## 2.2 CF-Prefix: specification for a new counterfactual dataset

Ordinary TTS corpora typically contain one recording per text and do not identify same-prefix/different-suffix conditional distributions. Build 600 template groups, each with four suffixes, three authorized speakers/voices, and two expressive realizations per complete sentence: 14,400 teacher utterances. Split template groups into train 360, validation 80, calibration 80, and test 80. All suffixes, voices, and seeds within a group stay together. This is a proposed collection, not an existing public dataset.

Cover four counterfactual types: phrase-boundary change, contrastive focus, number/abbreviation disambiguation, and discourse emotion trajectory. Separate lexical tone from emotional F0 in Mandarin. An LLM proposes candidates; humans verify complete-sentence plausibility and identical prefixes. Use two architecturally different full-context teachers for auxiliary labels to reduce single-teacher bias. Require native-speaker review of test cases; when collecting human audio, prioritize repeated calibration/test realizations over more synthetic training data.

The true distribution is exactly specified only in controlled experiments. Teacher samples define a teacher distribution, not human-optimal prosody. Retain teacher_id, teacher_seed, and prompt-audio provenance; report real and synthetic results separately. If the 80 calibration templates have correlated variants, T3 uses a template group as the exchangeable unit rather than pretending thousands of derivatives are independent calibration points.

```json
{"utt_id":"spk001_ch012_0004", "split":"train", "group_id":"book01_ch12",
 "audio_path":"/data/audio/u.wav", "sample_rate":24000,
 "text":"...", "phonemes":[12,7,19], "speaker_id":"spk001",
 "language":"en", "license":"CC-BY-4.0", "is_synthetic":false,
 "align_path":"/data/features/u.align.npz",
 "feature_path":"/data/features/u.npz",
 "token_arrival_ms":[0,45,92], "teacher_id":null,
 "reference_audio_id":null, "source_sha256":"..."}
```

Feature NPZ schema: linear[513,T], mel[80,T], duration[N], phone_mask[N], prosody[N,6], prosody_mask[N,6], token_to_phone[N,L], and llm_hidden[L,2048]. S3 adds audio_latent[64,T_A] and posterior scale; S4 adds codec_ids[L_A,16]. Cache metadata includes checkpoint hash, frontend version, sample rate, and normalization-statistics hash. Invalidate caches when any dependency changes.

## 2.3 Acoustic features and measurable targets

Remove corrupt files, clear transcript mismatches, and severe clipping. Preserve pauses; avoid wordwise cropping that destroys prosody. Apply utterance-level amplitude scaling, not framewise energy flattening. Initially train on 1–15-second samples; retain long utterances for streaming tests. Choose a fixed WORLD/pyworld or pYIN F0 extractor and never mix versions. Normalize using training statistics, either speaker-conditioned or global; do not use the complete target recording to normalize an unseen test speaker.

$$
Y_i=\left[\log(d_i h/f_s),\ \overline{\log F_0}_i,\ \operatorname{slope}_i(\log F_0),\ \log(E_i+\epsilon),\ v_i,\ \log(1+b_i)\right],
\tag{D1}
$$

**Symbols, dimensions, and conditions.** $Y_i\in\mathbb R^6$; h=256 samples/frame and $f_s=24000$ samples/s. $d_i$ counts frames, $E_i$ is RMS energy, $v_i\in[0,1]$ is voiced fraction, $b_i\ge0$ is following pause in seconds, and $\epsilon=10^{-5}$ stabilizes numerics. Mask F0 dimensions on unvoiced regions rather than inventing pitch; use a fixed mask and bounded normalization for the theorems.

First reproduce original MAS. Avoid treating early alignments as final labels during the first proposed 20k updates. Once stable, freeze a duration-label version for comparisons. Every method uses the same labels unless alignment itself is the experimental variable. MFA can independently audit boundaries; do not silently switch the label source between systems. Manually inspect at least 100 utterances as an engineering check, not a statistical guarantee.

## 2.4 Text-arrival traces and fair streaming evaluation

Store each test token and its wall-clock arrival. Controlled traces use 20/50/100 ms per token plus jitter and bursts. Real traces come from one frozen upstream LLM and are replayed identically for all TTS systems. Tokenization, G2P, and number normalization see only arrived prefixes. Report text-start-to-playable-first-chunk and stable-phoneme-to-playback latency; an internal first speech token is not the same as audible output.

$$
\operatorname{RTF}=t_{synth}/t_{audio},\quad L_i=t^{play}_i-t^{arrive}_i,\quad Q_{k+1}=\max(0,Q_k-\Delta t_k)+\ell_k.
\tag{D2}
$$

**Symbols, dimensions, and conditions.** All t, $L_i$, $Q_k$, $\Delta t_k$, and $\ell_k$ values are seconds. $Q_k$ is synthesized unplayed audio and $\ell_k$ the newly committed duration. RTF is dimensionless. Report mean, median, p95, first packet, word latency, and buffer underruns, not only overall RTF.

# 3. Shared implementation contracts, training schedule, and evaluation

## 3.1 Minimal module contracts

```python
# Interfaces to implement around the official VITS code.
# Tensor order is explicit; these are specification APIs, not upstream APIs.
encode_prefix(token_ids, arrival_mask) -> hidden[B,L,2048]
align_text(hidden, phone_spans) -> phone_cond[B,192,N]
prosody_prior(phone_cond, lm_pool) -> distribution_on_r[B,64]
prosody_posterior(real_prosody, mask) -> diagonal_gaussian[B,64]
duration_model(phone_cond, r, rhythm_state) -> samples[B,1,N]
expand_condition(phone_cond, duration) -> frame_cond[B,192,T]
acoustic_prior(frame_cond, r, history) -> distribution_on_z[B,192,T]
decode_chunk(z_chunk, decoder_state) -> wav[B,1,S_chunk], new_state
commit_policy(prefix_state, risk, buffer_seconds) -> WAIT | SPEAK
# Never call the waveform decoder for unobserved lexical content.
```

Adapt SynthesizerTrn from [R02], retaining the enc_q, enc_p, dp, flow, and dec names for auditability. Isolate additions in future_prosody.py, duration_state.py, risk.py, and streaming.py. Record git commit, model revision, pip freeze, CUDA/cuDNN, random seeds, and manifest hash. Separate Qwen3/tokenizer, VITS, and teacher environments when dependencies conflict; exchange cached NPZ/JSONL features.

## 3.2 Initial hyperparameters: proposed, not validated

```yaml
audio: {sample_rate: 24000, n_fft: 1024, hop_length: 256, n_mels: 80}
model: {phone_hidden: 192, latent_channels: 192, prosody_dim: 64, state_dim: 16}
llm: {id: Qwen/Qwen3-1.7B, frozen: true, hidden_size: 2048}
train:
  seeds: [17, 29, 43]
  optimizer: AdamW
  new_module_lr: 0.0001
  backbone_finetune_lr: 0.00002
  betas: [0.8, 0.99]
  weight_decay_new_modules: 0.01
  grad_clip: 1.0
  effective_batch: 32
  waveform_crop_samples: 8192
  precision: bf16
  density_math_precision: fp32
  validate_every_updates: 2000
  save_every_updates: 10000
  early_stop_validation_checks: 8
forecast: {offline_candidates: 8, online_candidates: 2, suffix_tokens: 16}
stream: {pending_phones: 8, max_wait_ms: 400, causal_decoder: true}
# Loss reductions must be explicit; probability derivations use sums.
# Do not interpret these settings as a guarantee of convergence.
```

Overfit 32 P0 utterances and pass single-batch shape checks before reproducing the full VITS baseline. Use a proposed cap of 300k updates with validation-based stopping. Train new prosody heads with the backbone frozen for 50k updates, then jointly for 50k at a smaller backbone learning rate. Each proposal defines additional phases. These are budget caps, not measured requirements. Validate densities, inverses, and KL in FP32 before mixed precision; do not naively evaluate logdet, variance, or ODE divergence in low precision.

Budget GPU resources from measured throughput rather than promising a fixed card count. Preflight frozen-LLM/cached-feature training on a 24–48 GB GPU; prefer staged freezing for large AudioVAE/DiT systems. After warmup, measure median step time across 200 updates: GPU-hours = updates × seconds/update × GPU count / 3600. This is an accounting rule, not a measurement made here. Report reused upstream-LLM and additional-TTS-LLM costs separately.

## 3.3 Required baselines and statistical tests

Required internal baselines: offline VITS, causal/chunked VITS, VITS2, hidden-state-only LLM fusion, fixed 0/1/2/4-word lookahead, lexical-entropy thresholds, and matched-budget random waiting. External references include CosyVoice, F5-TTS, VoXtream2, and X2Streaming-TTS. Never relabel an offline model as a true streaming baseline; separate comparisons with unmatched pretraining scale [R06,R13,R32,R35].

Report content (English WER/Mandarin CER, omissions/repetitions), speaker similarity, expression (duration/F0/boundary/semantic stress), runtime (D2), and probabilistic quality (NLL/CRPS/calibration). Use randomized blinded paired A/B listening on difficult prefixes and long text; the training teacher must not be the sole evaluator. A starting design is 200 test sentences with at least 10 independent judgments each, adjusted by pilot power analysis. Bootstrap speakers/listeners and texts hierarchically or use mixed effects, and run three seeds. These counts are study-design suggestions, not universal conference thresholds.

Preregister a meaningful improvement, primary metric, noninferiority margin, and compute budget. An internal go/no-go rule can require significantly better blind preference than hidden-state-only fusion at matched playable-first-packet latency, without worse content accuracy or p95 latency. Passing this rule does not guarantee acceptance. Do not cherry-pick significant hyperparameters/metrics; correct key multiple comparisons and release failed settings.

## 3.4 Unit tests and causal audits

Every system must pass: padding invariance; flow inverse error below 1e-5 in a small FP32 example; analytic/autodiff logdet agreement; MAS durations summing to frame count; samplewise invariance of committed audio when future text changes; full causal versus chunked decode parity under a fixed seed within stated tolerance; meaningful sensitivity to shuffled future candidates; and strict calibration/train/tuning separation. These are engineering checks, not probabilistic theorems.

# S1. Future-conditioned prosody diffusion: replacing a simple prosody prior

## S1.1 Motivation, component provenance, and hypothesis

**Motivation.** One text prefix can support several pause, intonation, and emphasis patterns. A single Gaussian prior may average distinct modes into an unsuitable realization. Borrow low-dimensional style diffusion from StyleTTS 2 [R04,R05], but not its weights without adaptation: our 64-dimensional block latent and prefix conditions differ. The rigorous version defines a DDPM probability chain [R17]; the existing StyleTTS 2 sampler is not automatically the ELBO written below.

**Falsifiable hypothesis.** At matched latent size, data, parameter budget, and playable-first-packet budget, diffusion covers counterfactual future-induced prosody modes better than a Gaussian/GMM and reduces conservative waiting in T3. MOS changes without distributional, coverage, or latency gains do not establish better uncertainty modeling.

```text
TRAIN:
full speech -> measured prosody -> q_eta(r0) --+-> VITS SDP/Flow/decoder
prefix -> Qwen3 -> future candidates --------+-> conditional diffusion
real waveform -> original VITS posterior ---+-> reconstruction/GAN

INFERENCE:
arrived prefix -> Qwen3/cache -> diffusion samples of r0
 -> duration distributions -> risk/commitment -> observed phones only
 -> VITS inverse flow -> causal waveform decoder
```

Retain enc_q, MAS, SDP, acoustic flow, and waveform generator. Replace the newly added simple ProsodyPrior and add ProsodyPosterior. Start with a four-block residual MLP denoiser, width 256, time embedding 128, and conditioning width 256; input/output are [B,64]. Decode r with phone features into [B,6,N] for local control instead of diffusing a high-frame-rate waveform.

## S1.2 Step-by-step probability derivation

$$
\begin{aligned}q_D(r_\tau\mid r_{\tau-1})&=N(\sqrt{a_\tau}r_{\tau-1},b_\tau I_R),\quad a_\tau=1-b_\tau,\\
\bar a_\tau&=\prod_{j=1}^{\tau}a_j,\qquad r_\tau=\sqrt{\bar a_\tau}r_0+\sqrt{1-\bar a_\tau}\epsilon.\end{aligned}
\tag{S1.1}
$$

**Symbols, dimensions, and conditions.** $R=64$ and $r_\tau,r_0,\epsilon\in\mathbb R^R$, with $\epsilon\sim N(0,I_R)$. $\tau=1,\ldots,D_{diff}$, with proposed training horizon $D_{diff}=100$. Noise variance $b_\tau\in(0,1)$ and $a_\tau,\bar a_\tau$ are scalars. Recursing Gaussian means/variances yields the second line.

$$
\begin{aligned}p_\psi(r_{0:D}\mid c_f)&=N(r_D;0,I_R)\prod_{\tau=1}^{D}N(r_{\tau-1};\mu_\psi(r_\tau,\tau,c_f),s_\tau^2I_R),\\
\mu_\psi&=a_\tau^{-1/2}\left(r_\tau-\frac{b_\tau}{\sqrt{1-\bar a_\tau}}\epsilon_\psi(r_\tau,\tau,c_f)\right).\end{aligned}
\tag{S1.2}
$$

**Symbols, dimensions, and conditions.** $c_f\in\mathbb R^{256}$ summarizes a prefix and candidate suffix. $\epsilon_\psi,\mu_\psi\in\mathbb R^{64}$; $s_\tau^2>0$ is reverse-kernel variance. In this section D means $D_{diff}$, not phoneme duration. Inference samples $r_D$ and iterates toward $r_0$.

$$
\begin{aligned}\widetilde b_\tau&=\frac{1-\bar a_{\tau-1}}{1-\bar a_\tau}b_\tau,\\
\widetilde\mu_\tau&=\frac{\sqrt{\bar a_{\tau-1}}b_\tau}{1-\bar a_\tau}r_0+\frac{\sqrt{a_\tau}(1-\bar a_{\tau-1})}{1-\bar a_\tau}r_\tau,\\
q_D(r_{\tau-1}\mid r_\tau,r_0)&=N(\widetilde\mu_\tau,\widetilde b_\tau I_R),\quad \tau\ge2.\end{aligned}
\tag{S1.3}
$$

**Symbols, dimensions, and conditions.** $\widetilde\mu_\tau\in\mathbb R^{64}$ and $\widetilde b_\tau$ is scalar. Complete the square in $q(r_\tau|r_{\tau-1})q(r_{\tau-1}|r_0)$. Handle $\tau=1$ with a separate positive-variance reconstruction kernel to avoid a degenerate covariance.

$$
\begin{aligned}\mathcal B_D(r_0,c_f)=&\mathbb E_q\log p_\psi(r_0\mid r_1,c_f)-\mathrm{KL}(q_D(r_D\mid r_0)\Vert N(0,I_R))\\
&-\sum_{\tau=2}^D\mathbb E_{q_D(r_\tau\mid r_0)}\mathrm{KL}(q_D(r_{\tau-1}\mid r_\tau,r_0)\Vert p_\psi(r_{\tau-1}\mid r_\tau,c_f))\\
\le&\log p_\psi(r_0\mid c_f).\end{aligned}
\tag{S1.4}
$$

**Symbols, dimensions, and conditions.** All terms are scalar. Insert $q_D/q_D$ into $\int p(r_{0:D})dr_{1:D}$, apply Jensen, and factor the forward chain backward by Bayes to obtain terminal KL, intermediate KLs, and first-step reconstruction. Use G8 for Gaussian KL; do not silently omit endpoint terms.

$$
\mathrm{KL}_\tau=C_\tau+\frac{b_\tau^2}{2s_\tau^2a_\tau(1-\bar a_\tau)}\mathbb E\|\epsilon-\epsilon_\psi(r_\tau,\tau,c_f)\|_2^2.
\tag{S1.5}
$$

**Symbols, dimensions, and conditions.** $C_\tau$ is scalar and independent of the mean network when $s_\tau$ is fixed. Subtract the S1.2 mean from the true posterior mean and substitute into the Gaussian KL mean-error term. Uniform unweighted MSE is a useful surrogate, not the exact weighted negative ELBO.

Define a finite-future model explicitly: sample and fix candidate set C from observed H; J indexes its M candidates with prior weights pi. Add CandidatePosterior producing $q_\kappa(J|r_0,H,C)$ to optimize a rigorous finite-mixture bound. Averaging Gaussian KLs is not the KL to a mixture.

$$
\begin{aligned}
\mathcal E_{S1}={}&\mathbb E_q\log p_\gamma(m\mid z)+\mathbb E_q\log p_\alpha(d^*\mid H,r_0)\\
&-\mathbb E_{q_\eta}\mathrm{KL}(q_\phi\Vert p_\theta)+\mathcal H(q_\eta)\\
&+\mathbb E_{q_\eta}\sum_{j=1}^{M}q_\kappa(j\mid r_0,H,C)\\
&\qquad\cdot\left[\mathcal B_D(r_0,c_j)+\log\pi_j-\log q_\kappa(j\mid r_0,H,C)\right].
\end{aligned}
\tag{S1.6}
$$

**Symbols, dimensions, and conditions.** $q_\eta$ is a positive-variance R-dimensional posterior; $\mathcal H$ is differential entropy and $\pi,q_\kappa\in\Delta^{M-1}$; $c_j\in\mathbb R^{256}$. This bounds $\log p(m,d^*|H,C)$: augment G4 with J and the diffusion path and collect log ratios. It is not the exact infinite-suffix marginal likelihood.

## S1.3 Data, training, inference, and resources

Use P0 for debugging and P1 for the main study, plus CF-Prefix training templates. Start with a 3:1 real/synthetic sampling ratio. Phase A: freeze VITS and train q_eta with an auxiliary prosody reconstructor for 20k updates to discourage collapse. Phase B: freeze posterior targets and train candidate-conditioned diffusion for 100k. Phase C: keep the vocoder fixed initially and condition SDP/prior on r for 50k joint updates. Phase D: enable streaming decoder/risk heads and fine-tune for 10k–30k validation-driven updates. Save stage checkpoints to avoid chasing a drifting posterior target.

At inference sample J and Gaussian r_D. Validate the probabilistic version with its full 100-step chain; report 4/8/16-step acceleration as separate approximations. Keep a block’s random seed fixed and update only uncommitted blocks when text arrives. Use guidance=1 for the strict probabilistic experiment; changed CFG kernels cannot inherit the original likelihood claim.

## S1.4 Required ablations, failure criteria, and novelty

Compare Gaussian, eight-component GMM, conditional flow, and diffusion priors at equal latent dimension. Ablate suffix candidates, one-future-only, shuffled suffixes, posterior prosody supervision, and fixed/random seeds. Plot NFE versus latency and listening/calibration. If GMM matches quality faster, diffusion is not justified. Larger variance is not accurate multimodality: measure held-out proper scoring rules, energy score, target-mode coverage, and spurious-mode frequency.

**Mathematical novelty: the diffusion ELBO is established; this is a rigorous derivation for a proposed conditional model.** Potential new contributions are acoustically calibrated sets with T3 or a validated future-equivalence model. **Readiness: not yet a completed top-conference submission.** Replacement alone is not a strong theoretical contribution; it needs a nontrivial additional result, strong baselines, and reproducible gains. Architecture/method references: [R01,R04,R05,R17,R22,R33,R34].

# S2. CosyVoice-style CFM: replacing the acoustic-prior transformation

## S2.1 Motivation, replacement boundary, and architecture

Pretrained CosyVoice CFM operates in Mel space with an 80-dimensional acoustic interface, whereas VITS transforms 192-channel acoustic latents [R07,R08]. Reuse causal DiT/CFM design, not allegedly plug-compatible weights. Keep VITS posterior, SDP, and decoder; replace flow with a conditional continuous normalizing flow. The question is whether continuous-time priors express future-prosody uncertainty better than discrete coupling under the same latency budget.

```text
TRAIN: waveform -> frozen VITS posterior -> z1[B,192,T]
       noise z0 + random ODE time + frame conditions -> causal DiT
       -> velocity regression; optional CNF density objective
INFER: arrived text -> future r -> SDP -> frame conditions
       -> Gaussian z0 -> ODE solver -> z1 -> VITS causal decoder
```

Start with a six-layer DiT vector field, width 384, six heads, FFN 1536, and conditioning from 192-dimensional frame features plus 64-dimensional r and speaker features. z input/output is [B,192,T_chunk], with only committed left context across blocks. An 80-channel CosyVoice input layer cannot be fixed by reshaping. Test causal masks/caches separately. Start at 32 latent frames, about 341 ms of audio, and ablate 16/64 frames.

## S2.2 CFM training and density derivation

$$
z_\tau=(1-\tau)z_0+\tau z_1,\quad u_\tau=z_1-z_0,\quad z_0\sim N(0,I_D),\quad z_1\sim q_\phi(z\mid x),
\tag{S2.1}
$$

**Symbols, dimensions, and conditions.** $D=192T_{chunk}$ and flattened $z_0,z_1,z_\tau,u_\tau\in\mathbb R^D$, with $\tau\sim U(0,1)$. Frame condition $c_f\in\mathbb R^{192\times T_{chunk}}$ is augmented by broadcast prosody/history. Endpoints here are independently paired; calling the pairing optimal transport requires an additional coupling procedure.

$$
\begin{aligned}\mathcal L_{CFM}&=\mathbb E\|v_\theta(z_\tau,\tau,c_f)-u_\tau\|_2^2,\\
v^*(z,\tau,c_f)&=\mathbb E[u_\tau\mid z_\tau=z,\tau,c_f],\\
\mathcal L_{CFM}(v)&=\mathbb E\|v-v^*\|^2+\mathbb E\|u_\tau-v^*\|^2.\end{aligned}
\tag{S2.2}
$$

**Symbols, dimensions, and conditions.** $v_\theta,v^*$ are D-vector fields and loss is scalar. Expand the square; the cross term vanishes by conditional expectation, giving the last line. Thus the regression optimum is the marginal velocity. This explains CFM but does not make its numerical value a KL or negative ELBO [R18].

$$
\frac{dz_\tau}{d\tau}=v_\theta(z_\tau,\tau,c_f),\qquad \partial_\tau p_\tau(z)=-\nabla_z\!\cdot(p_\tau(z)v_\theta(z,\tau,c_f)).
\tag{S2.3}
$$

**Symbols, dimensions, and conditions.** $p_\tau$ is a density on $\mathbb R^D$; $\nabla_z\cdot$ is divergence. The ODE defines transport and the second equation conserves probability mass. Assume sufficient regularity, integrability, and unique ODE solutions.

$$
\begin{aligned}\frac{d}{d\tau}\log p_\tau(z_\tau)&=\frac{\partial_\tau p_\tau}{p_\tau}+\nabla_z\log p_\tau\cdot v_\theta=-\nabla_z\cdot v_\theta,\\
\log p_1(z_1\mid c_f)&=\log p_0(z_0)-\int_0^1\operatorname{tr}(J_v(z_\tau,\tau,c_f))\,d\tau.\end{aligned}
\tag{S2.4}
$$

**Symbols, dimensions, and conditions.** $J_v\in\mathbb R^{D\times D}$. Apply the chain rule along the ODE, substitute S2.3, cancel the two gradient terms, and integrate. Evaluating density at a posterior z1 requires inverse ODE integration to recover z0. Stochastic trace estimation and numerical integration introduce estimation errors [R19].

$$
\operatorname{tr}(J_v)=\mathbb E_e[e^{\mathsf T}J_ve],\quad \mathbb E[ee^{\mathsf T}]=I_D,\qquad \mathcal L_z=\mathbb E_{q_\phi}[\log q_\phi(z_1\mid x)-\log p_1(z_1\mid c_f)].
\tag{S2.5}
$$

**Symbols, dimensions, and conditions.** $e\in\mathbb R^D$ can be an independent Rademacher vector; $e^TJ_ve$ is scalar. Use VJP/JVP without materializing the Jacobian, holding e fixed along an ODE path. The identity follows from $tr(J E[ee^T])=trJ$. Finite trace samples do not yield a deterministic per-example error guarantee.

Run two explicitly labeled versions. S2-P uses evaluable CNF density in place of G7, preserving G5 but incurring expensive training. S2-E freezes posterior/decoder, trains a latent prior with CFM, then fine-tunes reconstruction/GAN losses; it is an engineering generator, not an exact VAE objective. Report the different compute budgets.

## S2.3 Training and inference operations

Use the same P0/P1 splits. First obtain and freeze a stable VITS posterior checkpoint. Cache 192-channel posterior parameters, not just means, to retain stochasticity. Per-channel whitening is allowed, with its Jacobian constant included in density evaluation. Propose 150k CFM updates with encoder frozen. Use true durations for the first 50k, then aligned resampling of conditions for duration robustness; do not force one-to-one comparisons between unequal-length latents.

For S2-P, validate inverse ODE, density normalization, and gradients on D=8 synthetic data before real short chunks. Ablate 1e-3/1e-4/1e-5 tolerances and one/four trace vectors. S2-E uses Euler/Heun with 4/8/16 function evaluations, counting decoder and future-rollout latency. Only observed text and the pending block’s plan are used; a new block must not alter a previously emitted decoder state.

## S2.4 Comparisons, risks, and submission assessment

Compare original coupling flow, parameter-matched deeper coupling, Matcha/F5-style latent CFM, future conditioning on/off, and fixed/adaptive ODE steps. Report held-out prior NLL only where evaluable, posterior-reconstruction versus prior-synthesis gaps, and latency per NFE. If gains come from a larger DiT rather than future conditioning, narrow the claim. Typical failures are posterior drift, coarse-solver noise, boundary discontinuity, and expensive likelihood estimation.

**Novelty.** CNF density, trace estimation, and CFM regression are established; replacement alone is not a strong theory paper. A possible extension is a verifiable bound on conditional-path error under a compute budget that improves streaming step allocation; no new optimal-step theorem is proved here. **Readiness: incomplete; high risk and best treated as an independent generative-model project.** References: [R06–R08,R18–R20,R35,R36].

# S3. Paired replacement: VoxCPM2 AudioVAE and dual-clock VITS

## S3.1 Question, verified interfaces, and architecture

**Falsifiable hypothesis.** With the same text model and speech data, a pretrained continuous audio representation may improve the reconstruction ceiling, but only an explicit solution to the low-rate/phoneme-alignment mismatch can translate that ceiling into better generation. Replace enc_q and dec; retain phoneme duration modeling, conditional flows, and LLM future planning. Old 192-dimensional flow weights cannot be loaded directly into the new 64-dimensional latent space.

The verified VoxCPM2 configuration uses a 16 kHz encoder, 48 kHz decoder, and latent_dim=64. Encoder rates [2,5,8,8] and decoder rates [8,6,5,2,2,2] imply a 25 Hz AudioVAE latent clock. patch_size=4 is upper-model grouping, not the AudioVAE frame rate. The low-level encoder returns mu/logvar, whereas the convenient encode interface returns mu only. A strict VAE must inspect and use a stochastic posterior rather than relabel a deterministic vector as a continuous finite-KL density [R09,R10].

```text
TRAIN: waveform -> resample 16 kHz -> frozen AudioVAE encoder -> q_A(u)
       waveform -> 24 kHz Mel -> high-rate aligner -> phone durations
       observed text -> Qwen states -> phone features + future r
       durations -> interval-overlap adapter -> 25 Hz condition
       conditional flow prior <-> q_A(u) -> AudioVAE decoder -> waveform
INFER: observed text -> r -> SDP -> high-rate durations -> overlap adapter
       Gaussian base -> new 64-channel flow inverse -> u -> streaming AudioVAE decoder
```

| Location | Operation | Tensor contract |
|---|---|---|
| Acoustic posterior | Replace with AudioVAE encoder | $\mu_A,\ell_A:B\times64\times T_A$ |
| Duration/alignment | Keep a high-rate clock; do not force MAS on low-rate latents | $d:B\times1\times N$; $O:B\times T_A\times N$ |
| Prior/flow | Retrain; project conditions 192-to-64 | $c_A:B\times192\times T_A$; $u:B\times64\times T_A$ |
| Waveform generation | Replace with paired AudioVAE decoder | $B\times64\times T_A\to B\times1\times S_{48}$ |

## S3.2 Deriving the dual-clock transformation

$$
\Delta_h=256/24000,\quad \Delta_A=640/16000=1920/48000=0.04,\qquad T_A=\lceil T\Delta_h/\Delta_A\rceil.
\tag{S3.1}
$$

**Symbols, dimensions, and conditions.** $\Delta_h,\Delta_A$ are measured in seconds; the clocks are 93.75 and 25 frames/s. $T_A$ is the low-rate length. Actual encoder padding may add boundary frames; cache actual output lengths and valid masks instead of assuming this expression equals every API output length.

$$
\begin{aligned}b_i&=\Delta_h\sum_{j\le i}d_j,\quad I_i=[b_{i-1},b_i),\quad J_t=[(t-1)\Delta_A,t\Delta_A),\\ O_{ti}&=\frac{|J_t\cap I_i|}{|J_t\cap[0,b_N)|},\qquad \sum_iO_{ti}=1,\qquad c_A=hO^{\mathsf T}.\end{aligned}
\tag{S3.2}
$$

**Symbols, dimensions, and conditions.** $b_0=0$ and $b_i$ are seconds. $I_i,J_t$ are temporal intervals and vertical bars denote interval length. $O\in\mathbb R_+^{T_A\times N}$ and $c_A\in\mathbb R^{192\times T_A}$. Evaluate only nonzero denominators. Phone intervals partition each valid J, yielding a row sum of one. Soft overlap allows several short phones inside one latent frame, avoiding the invalid MAS path when $T_A<N$.

At inference, form intervals from integer high-rate durations and map them to the low-rate clock. Padding and cropping control length; do not arbitrarily warp one valid example onto inconsistent timelines during training. A differentiable soft-duration overlap approximation is possible, but must be checked against the integer deployment path. The adapter transfers conditions; it does not guarantee that a 25 Hz representation distinguishes every short phone.

## S3.3 VAE and new-prior density derivation

$$
\begin{aligned}q_A(u\mid y)&=\prod_{c=1}^{64}\prod_{t=1}^{T_A}\mathcal N(u_{ct};\mu_{A,ct},\exp(\ell_{A,ct})),\\ u&=\mu_A+\exp(\ell_A/2)\odot\epsilon,\quad \epsilon\sim\mathcal N(0,I_{64T_A}).\end{aligned}
\tag{S3.3}
$$

**Symbols, dimensions, and conditions.** $u,\mu_A,\ell_A$ all have shape $64\times T_A$. $\ell_A$ is log variance, hence division by two in reparameterization. Independence is the conditional diagonal variational approximation, not actual acoustic independence. Inspect the loaded logvar branch, its training status and range. A mean-only cache cannot evaluate this density.

$$
\begin{aligned}v&=f_\theta(u;c_A,r,s),\\ \log p_\theta(u\mid c_A,r,s)&=\log\mathcal N(v;\mu_p,\operatorname{diag}\sigma_p^2)+\log|\det J_f(u)|,\\ \log p(m\mid c_A,r,s)&=\log\int q_A(u\mid y)\frac{p_\gamma(m\mid u)p_\theta(u\mid c_A,r,s)}{q_A(u\mid y)}du\\ &\ge\mathbb E_{q_A}\log p_\gamma(m\mid u)-\mathrm{KL}(q_A\Vert p_\theta).\end{aligned}
\tag{S3.4}
$$

**Symbols, dimensions, and conditions.** $v,\mu_p,\sigma_p$ have shape $64\times T_A$. $s$ is a fixed-interface speaker condition of dimension $C_s$, read at runtime and projected; speaker embeddings from different systems are not interchangeable. $J_f$ is $64T_A\times64T_A$. Change of variables gives the second line; Jensen and log-ratio grouping give the last. Use G6 for the Mel likelihood, centered on the AudioVAE decoder output.

The strict version retains posterior stochasticity and evaluates the new prior density. An engineering version may regress fixed mu latents, but then it does not have the continuous-posterior KL in S3.4. Report them separately. A decoder can instead define a Gaussian waveform mean, yielding a normalized waveform likelihood, but that changes the reconstruction loss; spectral/GAN losses do not automatically become that Gaussian NLL.

## S3.4 Data, training stages, and online procedure

Use P1/P1-ZH. Freeze AudioVAE and first measure the real-audio reconstruction ceiling: encode resampled 16 kHz audio, decode at 48 kHz, and compare at a common 24 kHz rate. Separately evaluate high-frequency quality on native 48 kHz data. Upsampling 24 kHz recordings does not create genuine high-frequency supervision. Obtain d* from an independent high-rate forced aligner or a stable VITS aligner; reject T<N and abnormal-silence examples.

Stage A: cache mu/logvar, d*, phones, and masks. Stage B: freeze AudioVAE/Qwen; train the 64-channel flow, prior, and overlap projection for a proposed 100k updates using true durations. Stage C: add future-r and high-rate SDP for 50k updates, keeping true/predicted-duration paths distinct instead of comparing unequal-length frames. Stage D: only after stabilizing the prior, try unfreezing the last two decoder stages at 1e-5 for at most 20k updates, comparing a fully frozen version. Do not simultaneously unfreeze the encoder and shift the latent distribution.

Online, confirm a committable phone span, sample d, construct O, sample and invert the low-rate prior, and call a cache-validated streaming AudioVAE decoder. One 25 Hz latent corresponds to about 40 ms, but first-playable latency additionally includes receptive field, buffering, and computation. A streaming_decode API alone does not establish zero right context; test full versus chunked decoding parity.

## S3.5 Experiments, failure criteria, and novelty audit

Ablations: original VITS; frozen AudioVAE with a repetition adapter; overlap adapter; mean versus stochastic posterior; native 25 Hz versus controlled regrouping; semantic auxiliary loss; frozen versus tuned decoder. Measure reconstruction/prior WER and SIM gaps, short-phone omissions, alignment failures, latent NLL, latent bandwidth, and latency. Improved reconstruction with degraded prior synthesis is not a TTS improvement.

Existing AudioVAEs, hierarchical generation, and Semantic-VAE make a codec swap insufficiently novel [R09,R37]. Potential contributions are dual-clock alignment compatibility, a reproducible predictability/compression study, and T4’s prefix-consistent conditional density. No lossless-phone-preservation or optimal-representation theorem is established here. **Readiness: unvalidated; a method submission requires stable gains across data scales, difficult short-phone cases, and compute-matched controls.**

# S4. Discrete speech semantics with continuous VITS: Qwen3-TTS components

## S4.1 Motivation, replacement scope, and one duration authority

This proposal tests whether low-rate speech tokens improve VITS generalization. Reuse Qwen3-TTS-Tokenizer-12Hz: its name corresponds to 12.5 Hz, 16 codebooks, and 2048 entries per codebook, not the generic Qwen text tokenizer [R11,R12]. Retain VITS enc_q, continuous flow, and waveform decoder; add a speech-token predictor and token-to-frame adapter. Semantic supervision of the first codebook does not imply perfect disentanglement from timbre or prosody.

Choose who owns duration. In S4-D, VITS SDP alone determines high-rate T and token length L_A; the speech LM generates at the prescribed length. In S4-L, the original speech LM/EOS owns duration and independent SDP length control is removed, retaining alignment only as auxiliary supervision. Free token-LM EOS and forced SDP length cannot coexist without an explicit conflict rule. Implement S4-D first; S4-L is a comparator.

```text
OFFLINE: audio -> frozen Qwen speech tokenizer -> a*[B,L_A,16]
TRAIN: phones + observed text/LLM -> future r -> SDP -> length L_A
       text/length/r -> speech-token LM -> token distributions
       teacher tokens or sampled tokens -> frame adapter -> VITS flow condition
       VITS posterior(audio) -> KL/reconstruction/GAN
INFER: stable phones -> SDP length -> token LM with length budget
       generated tokens -> adapter -> continuous VITS prior -> waveform
```

| Variable | Shape | Meaning |
|---|---|---|
| $a$ | $B\times L_A\times16$, integer | Residual-codebook indices per token frame, values 0…2047 |
| $e_a$ | $B\times L_A\times256$ | New token-adapter embeddings; 256 is a proposed width |
| $\Lambda$ | $B\times L_A\times16\times2048$ | Conceptual logits; evaluate incrementally to avoid storing the full tensor |
| $c_z$ | $B\times192\times T$ | Phone, future-r, and token conditions fused on the high-rate timeline |

## S4.2 Deriving the discrete–continuous ELBO

$$
L_A=\left\lceil12.5\,\Delta_h\sum_i d_i\right\rceil,\qquad p_\omega(a\mid H,r,d)=\prod_{t=1}^{L_A}\prod_{q=1}^{16}p_\omega(a_{tq}\mid a_{<t},a_{t,<q},H,r,d).
\tag{S4.1}
$$

**Symbols, dimensions, and conditions.** $L_A$ is token-frame count and q indexes codebooks. Every conditional distribution is a 2048-way softmax. The time-major/within-frame chain is a proposed factorization, not a claim to reproduce every detail of Qwen3-TTS dual-track computation. Convert units with $\Delta_h=256/24000$.

$$
\begin{aligned}p(a,z,m\mid H,r,d)&=p_\omega(a\mid H,r,d)p_\theta(z\mid a,H,r,d)p_\gamma(m\mid z),\\q(a,z\mid y)&=\delta_{a^*(y)}(a)q_\phi(z\mid y,a^*),\\p(m\mid H,r,d)&=\sum_a\int p(a,z,m\mid H,r,d)dz\\&\ge\int p(a^*,z,m\mid H,r,d)dz.\end{aligned}
\tag{S4.2}
$$

**Symbols, dimensions, and conditions.** $\delta$ is a discrete point mass, $a^*$ the frozen tokenizer output, and $z\in\mathbb R^{192\times T}$. Every term in the discrete sum is nonnegative, so retaining the a* term gives a lower bound. Insert the q_phi ratio only for the continuous z integral, rather than incorrectly treating a point-mass posterior as an importance distribution covering all a.

$$
\begin{aligned}\log p(m\mid H,r,d)&\ge\log p(a^*,m\mid H,r,d)\\&\ge\mathbb E_{q_\phi}\log p_\gamma(m\mid z)+\log p_\omega(a^*\mid H,r,d)\\&\quad-\mathrm{KL}(q_\phi(z\mid y,a^*)\Vert p_\theta(z\mid a^*,H,r,d)),\\\mathrm{KL}(\delta_{a^*}\Vert p_\omega)&=-\log p_\omega(a^*\mid H,r,d).\end{aligned}
\tag{S4.3}
$$

**Symbols, dimensions, and conditions.** The first step retains one nonnegative term from the discrete sum. The second applies Jensen to the z integral and groups log ratios into KL. A discrete point mass has zero entropy, so token KL equals token cross-entropy. A continuous Dirac generally does not have the same finite-KL property.

$$
e_{a,t}=\sum_{q=1}^{16}E_q[a_{tq}],\qquad c_z=\operatorname{Conv}_{1\times1}([h^{frm};(e_a^{\mathsf T}O_{tok\to frm});r\mathbf1_T^{\mathsf T}]).
\tag{S4.4}
$$

**Symbols, dimensions, and conditions.** $E_q\in\mathbb R^{2048\times256}$; $e_a^T\in\mathbb R^{256\times L_A}$. $O_{tok\to frm}\in\mathbb R_+^{L_A\times T}$ is interval-overlap normalized by high-rate frame. $r\mathbf1_T^T\in\mathbb R^{64\times T}$. Concatenation has shape $512\times T$ and a 1×1 convolution outputs $192\times T$. These new embeddings do not identify codec codebook vectors with VITS z.

## S4.3 Data, reusable weights, and staged training

Use P1 and a fixed P2 subset, freezing the tokenizer and caching source rates, valid lengths, and codec version. Compare S4-small, a newly trained six-layer width-512 causal token predictor, with S4-pre, transferred speech-LM weights matched to that tokenizer and teacher-forced under their original input protocol, then adapted with length/r adapters or LoRA. S4-small transfers a pretrained tokenizer, not a pretrained speech LM.

Stage 1: train VITS adapter/prior under true tokens for 50k updates and measure the teacher-token ceiling. Stage 2: independently optimize token CE until validation stops improving, capped at 200k updates; report frame/codebook CE separately. Stage 3: increase sampled-token conditioning probability from zero to 0.5 over 50k updates to reduce exposure mismatch. Do not backpropagate ordinary gradients through discrete samples. Straight-through/Gumbel alternatives must be labeled biased estimators rather than exact discrete-ELBO gradients.

Inference obtains a stable pending span, d, and L_A. The token predictor generates exactly the allocated frame budget, with EOS not terminating prematurely. A checkpoint not trained for this constraint must be adapted and tested for content completeness; external audio truncation is not a valid substitute. Then condition the flow with S4.4. Small token groups can be decoded incrementally, with separate causal-cache tests.

## S4.4 Controls and publication boundary

Required controls: token-free VITS; first codebook only; all 16; teacher versus generated tokens; S4-L without independent SDP; full original Qwen3-TTS; and the same token predictor with a conventional Mel/codec decoder. Quantify redundancy: does SDP add duration control after tokens are known? Report AR-token compute, memory, and first-playable latency rather than hiding poorer responsiveness behind average RTF.

**Theory status.** The discrete–continuous ELBO and its CE reduction are established tools. New value requires an identifiable division of information or an empirically justified dual-clock control mechanism. A backend swap without compute, content, or controllability benefits is a weak method contribution. This route is heavier than S6 and should not be trained alongside every other replacement. References: [R01,R11,R12,R20].

# S5. Training-only transfer: a WavLM perceptual discriminator

## S5.1 Motivation, structure, and gradient path

This is the lowest-risk component proposal but the weakest standalone novelty. Following StyleTTS 2, feed pretrained WavLM representations to a new discriminator head to supplement waveform-period discrimination. This does not automatically understand target text or guarantee removal of omissions; independent content evaluation remains necessary [R04,R05,R15]. Keep enc_p, enc_q, SDP, flow, and generator unchanged; add only a training branch that disappears at inference.

```text
real audio --------------------> differentiable 16 kHz resampler -> WavLM -> head D
text -> VITS -> generated audio -> differentiable 16 kHz resampler -> WavLM -> head D
WavLM parameters: frozen
Discriminator step: detach generated waveform; update head D only
Generator step: freeze D parameters; keep graph through WavLM and resampler to VITS
```

Start with WavLM Base as a resource baseline, then test Large; Large has hidden width 1024, with layer counts and valid lengths read from the loaded config. Select layers 3/6/9/12 or valid corresponding layers, apply per-layer LayerNorm and 1×1 projections to width 128, masked-pool, and use a two-layer MLP for a scalar score. Layer selection is a proposal to fix before comparison. WavLM temporal downsampling is not assumed to align exactly with Mel or phone frames.

## S5.2 Adversarial objectives, dimensions, and differentiation

$$
F_\ell(y)=\operatorname{WavLM}_\ell(\mathcal R_{16k}(y)),\qquad D_\xi(y)=C_\xi(\{F_\ell(y)\}_{\ell\in\mathcal S}).
\tag{S5.1}
$$

**Symbols, dimensions, and conditions.** $\mathcal R_{16k}$ is a differentiable resampler. $F_\ell\in\mathbb R^{B\times L_w\times C_w}$; $\mathcal S$ is a fixed layer-index set. The model determines $L_w,C_w$, and $D_\xi\in\mathbb R^B$. $\xi$ contains only new-head parameters; WavLM weights stay fixed.

$$
\begin{aligned}\mathcal L_D&=\mathbb E_y[(D_\xi(y)-1)^2]+\mathbb E_{\hat y}[D_\xi(\operatorname{sg}(\hat y))^2],\\\mathcal L_{G,SLM}&=\mathbb E_{\hat y}[(D_\xi(\hat y)-1)^2],\\\mathcal L_G&=-\mathcal E+\lambda_{adv}\mathcal L_{G,VITS}+\lambda_{fm}\mathcal L_{fm}+\lambda_{SLM}\mathcal L_{G,SLM}.\end{aligned}
\tag{S5.2}
$$

**Symbols, dimensions, and conditions.** $\operatorname{sg}$ denotes stop-gradient and $\hat y=G_\gamma(z)$. Losses are scalar; $\mathcal E$ is the probabilistic bound in G5, while other terms are auxiliary objectives. Least-squares discrimination is a design choice; a hinge comparison is possible but should not be mislabeled as exact reproduction of all upstream losses.

$$
\nabla_\gamma\mathcal L_{G,SLM}=2(D_\xi(\hat y)-1)\,J_G^{\mathsf T}J_{\mathcal R}^{\mathsf T}J_F^{\mathsf T}\nabla_F C_\xi.
\tag{S5.3}
$$

**Symbols, dimensions, and conditions.** This is the single-example chain rule: $J_G$ maps parameter perturbations into waveform changes; subsequent Jacobians follow waveform, resampling, and features. Freezing WavLM parameters does not make $J_F=0$. A no_grad wrapper around WavLM during the generator step incorrectly cuts the needed gradient. Set parameter requires_grad_(False) while retaining the input graph.

S5.2 is a hybrid variational/adversarial objective, not the same likelihood derived from G5. Adding WavLM does not establish a tighter ELBO. Gradient flow through a frozen backbone follows the chain rule, not a new theorem.

## S5.3 Executable training design, data, and ablations

Start from the same stable VITS checkpoint on P0/P1. Train the head for 5k updates on frozen real/generated caches, then alternate D/G for 50k updates. Proposed settings: 1:1 D:G, head lr=1e-4, generator lr=2e-5; ramp lambda_SLM from zero to 0.1 and compare 0.01/0.3 to avoid abruptly dominant adversarial gradients. Propose 2–4 s audio crops, profiling memory before fixing the batch. Match total updates/compute across candidates.

Evaluate the original discriminator, SLM-only, parallel discriminators, layer subsets, a random same-architecture backbone, and frozen versus unfrozen WavLM. Do not prove improvement only with WavLM-derived metrics: include independent ASR, a different-family speaker encoder, and human naturalness/expressiveness preferences. Monitor gradient norms, clipping/silence failures, and adversarial noise textures. Inference omits the new head, so separate training and inference costs.

**Novelty/readiness.** S5 alone does not meet a strong-method contribution expectation because direct precedent exists. It may improve S1/S3/S6 perceptually or stabilize training, but needs its own ablation. An independent paper would require a new provable perceptual constraint or an account of transferable naturalness feedback; no such new property is proved here.

# S6. Stateful, future-conditioned stochastic duration modeling

## S6.1 Relationship to VoXtream2 and IndexTTS2

This is the preferred lightweight streaming modification. Retain original VITS flow-SDP rather than substitute token-count control; motivated by VoXtream2’s duration-state work, add a small stochastic rhythm state rho. IndexTTS2 token-count control is a related comparator, not a drop-in phoneme SDP [R13,R14]. The Gaussian transition below is a new design specification, not a claim to reproduce either paper’s implementation.

Hypothesis: some local duration variation is explained by cross-chunk rhythm memory, while another part depends on future context. Separating them may improve incremental control over independent noise or a single speaker embedding. Test a 2×2 state/future ablation; do not assume disentanglement has already succeeded.

```text
observed text / Qwen prefix states -> semantic adapter -> phone h
future prosody head -> r_k[64] ----------------------+
previous rhythm state rho_{k-1}[16] -> transition --> rho_k[16]
(h, r_k, rho_k) -> original flow-SDP with new condition -> duration distribution
                         -> acoustic-risk probe -> WAIT or SPEAK
SPEAK -> commit d and state -> duration expansion -> VITS flow -> causal decoder
WAIT  -> update text belief only; do not falsely advance committed acoustic state
```

## S6.2 Module specification and state transition

$$
\begin{aligned}\bar h_k&=\operatorname{MaskedMean}(h_k),\\ \mu_{\rho,k}&=\tanh(W_\rho\rho_{k-1}+U_h\bar h_k+U_r r_k+b_\rho),\\ \sigma_{\rho,k}&=\operatorname{softplus}(V[\rho_{k-1};\bar h_k;r_k]+b_\sigma)+\sigma_{min},\\ p_\beta(\rho_k\mid\rho_{k-1},H_k,r_k)&=\mathcal N(\mu_{\rho,k},\operatorname{diag}\sigma_{\rho,k}^2).\end{aligned}
\tag{S6.1}
$$

**Symbols, dimensions, and conditions.** $\bar h_k\in\mathbb R^{192}$, $\rho_k\in\mathbb R^{16}$, $r_k\in\mathbb R^{64}$. Shapes: $W_\rho:16\times16$, $U_h:16\times192$, $U_r:16\times64$, $V:16\times272$, and biases are 16-dimensional. Proposed $\sigma_{min}=10^{-3}$. The spectral constraint $\|W_\rho\|_2\le0.9$ bounds mean recurrence under fixed inputs, not automatic contraction of the entire feedback loop.

$$
c^{dur}_{ki}=\operatorname{sg}(h_{ki})+A_r r_k+A_\rho\rho_k,\qquad d_k\sim p_\alpha(d_k\mid c^{dur}_k,s).
\tag{S6.2}
$$

**Symbols, dimensions, and conditions.** $h_{ki},c^{dur}_{ki}\in\mathbb R^{192}$; $A_r:192\times64$ and $A_\rho:192\times16$. Inject state/future conditioning after the original detach, or duration loss cannot train the new branches. Retain SDP dequantization u, augmentation v, and invertible duration flow; use the likelihood bound G9.

$$
\begin{aligned}
&p(r_{1:K},\rho_{0:K},d_{1:K},z_{1:K},m_{1:K}\mid H_{1:K})\\
&\quad=p(\rho_0)\prod_{k=1}^K\Big\{p_\psi(r_k\mid H_k)\\
&\qquad\cdot p_\beta(\rho_k\mid\rho_{k-1},H_k,r_k)
 p_\alpha(d_k\mid H_k,r_k,\rho_k)\\
&\qquad\cdot p_\theta(z_k\mid H_k,r_k,\rho_k,d_k)
 p_\gamma(m_k\mid z_{\le k},H_k)\Big\}.
\end{aligned}
\tag{S6.3}
$$

**Symbols, dimensions, and conditions.** $K$ counts committed speech blocks, not LLM tokens. $\rho_0$ can be standard Gaussian or a known fixed zero state; record the chosen variant. $d_k\in\mathbb N_+^{N_k}$ and $z_k\in\mathbb R^{192\times T_k}$. These are kernels under a prescribed causal input path. If H contains feedback generated from model outputs, this is not an arbitrary observational-joint factorization.

## S6.3 Sequential ELBO and posterior definition

$$
q= q_0(\rho_0)\prod_{k=1}^K q_\eta(r_k\mid y,c_{full})q_\nu(\rho_k\mid\rho_{k-1},r_k,y,c_{full})q_\phi(z_k\mid x_k,r_k,\rho_k,d_k^*).
\tag{S6.4}
$$

**Symbols, dimensions, and conditions.** $q_\nu$ outputs 16-dimensional mean/log variance; a training-only two-layer bidirectional GRU over six-dimensional phone prosody followed by pooling is one implementation. $q_\eta$ outputs 64-dimensional parameters. If $q_0=p(\rho_0)$, the initial KL is zero. Full-audio information is legal in the training posterior, which must be removed at inference.

$$
\begin{aligned}\mathcal E_{seq}=&-\mathrm{KL}(q_0\Vert p(\rho_0))\\ &+\sum_{k=1}^K\mathbb E_q\big[\log p_\gamma(m_k\mid z_{\le k},H_k)+\log p_\alpha(d_k^*\mid H_k,r_k,\rho_k)\big]\\ &-\sum_k\mathbb E_{q(<k)}\mathrm{KL}(q_\eta(r_k)\Vert p_\psi(r_k\mid H_k))\\ &-\sum_k\mathbb E_{q(<k),q_\eta}\mathrm{KL}(q_\nu(\rho_k\mid\cdot)\Vert p_\beta(\rho_k\mid\rho_{k-1},H_k,r_k))\\ &-\sum_k\mathbb E_{q(<k),q_\eta,q_\nu}\mathrm{KL}(q_\phi(z_k\mid\cdot)\Vert p_\theta(z_k\mid H_k,r_k,\rho_k,d_k^*)).\end{aligned}
\tag{S6.5}
$$

**Symbols, dimensions, and conditions.** $q(<k)$ includes posterior distributions of all relevant earlier states. Each KL integrates the current variable; its outer expectation must remain. Proof: multiply/divide the integral of S6.3 by q, apply Jensen, expand the log product into a blockwise sum, and collect each current-variable KL. Use G8 for Gaussian rho KL, G7 for z density, and G9 for the duration lower bound.

Fixing previous states to teacher means optimizes a teacher-forced conditional surrogate, not the complete expectation in S6.5. Distinguish posterior sampling, teacher-mean approximation, and prior-rollout fine-tuning. Beta-KL, free bits, and KL warmup may stabilize training, but cease to be the unweighted exact ELBO; log both raw bound estimates and the optimized surrogate.

## S6.4 Data, training, and inference algorithm

P0 validates local stability only. P1/P1-ZH training must preserve adjacent-sentence order within recordings; randomly concatenating speakers does not establish cross-sentence memory. Without neighboring context, study within-utterance states only. CF-Prefix supplies future-impact supervision; real-audio MAS durations train the state posterior. Randomize chunk boundaries at 4/8/16 phones while preserving one underlying timeline.

Stage A: freeze VITS and train q_eta, q_nu, and duration/pause auxiliary predictors for 30k updates. Stage B: unfreeze SDP conditioning and transition for 50k updates, warming rho KL from zero to one over the first 10k and fixing the variance floor. Stage C: train 50k updates with prior-state rollouts, increasing their share from 25% to 75%; keep a separate teacher branch for variational diagnostics. Stage D: only after reconstruction stabilizes, unfreeze prior/flow for 20k–50k updates. Every checkpoint passes prefix-leakage, positive-duration, and accumulated-time tests.

```text
state = zero_or_sampled_rho0
for observed_prefix in text_arrival_stream:
    phones = frontend.stable_unspoken_phones(observed_prefix)
    r_dist = future_head(prefix_cache)
    proposed_state = transition(state, phones, r_dist)
    risk = calibrated_probe(phones, r_dist, proposed_state)
    if policy(risk, audio_buffer, deadline) == WAIT:
        continue  # do not advance committed duration state
    d = sdp.sample(phones, r_dist, proposed_state)
    z = acoustic_prior.sample(phones, d, r_dist, proposed_state)
    emit(causal_decoder(z, committed_cache))
    state = proposed_state
    commit_phone_span_and_durations()
```

For duration-risk estimation, add a normalized categorical probe over 1…400 frames plus an overflow class. It does not replace SDP; it provides evaluable risk probabilities. Overflow represents all longer durations and needs a deployment rule. Calibrate probe-to-SDP and probe-to-waveform discrepancies; do not identify auxiliary probabilities with the exact SDP distribution.

## S6.5 Ablations, failure criteria, theory, and publication positioning

Required ablations: neither state nor future, state only, future only, both; deterministic/random rho; contraction off/on; teacher forcing/prior rollout; real versus shuffled state; fixed/dynamic chunks; L1 duration regression versus original flow-SDP. Evaluate duration NLL proxies, rate jumps, pauses, content completeness, cross-chunk F0/style drift, and p95 playback latency. Oversmoothing, ignored rho, and speaker leakage are failures; low variance is not naturalness.

**Theory/novelty.** State-space VAEs and Gaussian KL are established; S6 is a plausible but unvalidated structural change. T2’s acoustic transport bounds, T3’s trajectory-risk guarantee, or T5’s computation-aware stopping provide a more distinctive question. Show that gains are not merely extra parameters or regularization, and validate across languages or arrival rates. **Readiness: incomplete; among the six components, this is the most controllable first research system.**

# T1. Causal prosody distortion and estimable future impact

## T1.1 Theoretical question, architecture, and target

This direction separates model inadequacy from information that has not arrived, rather than labeling another network block as theory. Fix a pending span of n* phones and its q=6n*-dimensional prosody Y. Prefix observations generate sigma-algebra F_k; after the suffix W arrives, information is F_k joined with sigma(W). Neither includes magically available reference audio. A full-context teacher defines an evaluation target, not an online input.

```text
paired full-context recordings -> aligned current-span target Y
same observed prefix -> Qwen/future head -> mu_prefix
prefix + actual suffix -> full-context predictor -> mu_full
compare irreducible future gap, model gap, and measured delay frontier
at inference: prefix-risk estimator -> wait policy -> unchanged S6/VITS backend
```

Use independent CF-Prefix template groups with a fixed speaker and identical current phone span, varying only the suffix. At least two recordings/teacher samples per suffix permit an initial separation of expressive randomness; native listeners must validate a real-recording subset. Suffix length does not change Y dimension because the same pending span is evaluated. Fix voiced-mask rules rather than selecting easier F0 frames separately for each model.

## T1.2 Proposition 1: prefix Bayes-risk decomposition, full proof

$$
\mu_k=\mathbb E[Y\mid\mathcal F_k],\quad \mu_*=\mathbb E[Y\mid\mathcal F_k,W],\quad R_k(f)=\mathbb E\|Y-f(\mathcal F_k)\|_2^2.
\tag{T1.1}
$$

**Symbols, dimensions, and conditions.** $Y,\mu_k,\mu_*,f\in\mathbb R^q$, with proposed q=48 for n*=8. $R_k$ is scalar. Assume Y is square-integrable and f is measurable using only F_k. Here f is a prosody point predictor, not a complete measure of stochastic speech quality.

$$
\begin{aligned}Y-f&=(Y-\mu_k)+(\mu_k-f),\\\mathbb E\|Y-f\|^2&=\mathbb E\|Y-\mu_k\|^2+\mathbb E\|\mu_k-f\|^2\\&\quad+2\mathbb E[(Y-\mu_k)^{\mathsf T}(\mu_k-f)],\\\mathbb E[(Y-\mu_k)^{\mathsf T}(\mu_k-f)\mid\mathcal F_k]&=(\mathbb E[Y\mid\mathcal F_k]-\mu_k)^{\mathsf T}(\mu_k-f)=0.\end{aligned}
\tag{T1.2}
$$

**Symbols, dimensions, and conditions.** Norms are Euclidean. The third line moves the F_k-measurable vector mu_k−f outside conditional expectation. Taking total expectation eliminates the cross term; neither Gaussianity nor linear networks is needed.

$$
\begin{aligned}R_k^*&=\inf_fR_k(f)=\mathbb E\operatorname{tr}\operatorname{Cov}(Y\mid\mathcal F_k),\\\Gamma_k&=R_k^*-R_*^*=\mathbb E\|\mu_*-\mu_k\|^2\ge0,\\R_k(f)-R_*^*&=\Gamma_k+\mathbb E\|\mu_k-f\|^2.\end{aligned}
\tag{T1.3}
$$

**Symbols, dimensions, and conditions.** $R_*^*$ is minimum risk for the same squared loss with the full suffix visible. Cov is q×q. T1.2’s nonnegative second term yields the first line. Repeat the orthogonal decomposition with Y−mu_k=(Y−mu*)+(mu*−mu_k) and use iterated expectation for the second; substitution gives the third. Gamma is missing-information loss and the final term is learnable prediction error.

$$
P(W=1)=P(W=2)=\tfrac12,\quad Y\mid W=j=y_j\quad\Longrightarrow\quad\Gamma=\tfrac14\|y_1-y_2\|^2.
\tag{T1.4}
$$

**Symbols, dimensions, and conditions.** $y_1,y_2\in\mathbb R^q$ are deterministic suffix-specific targets. The prefix mean is their midpoint; each deviation is half the difference, yielding the one-quarter factor. If suffixes do not change current prosody, the bound is zero.

Boundary: T1.3 applies classical conditional-expectation orthogonality; it is not a first discovery. Independent random generation adds sampling variance under squared error, but this does not make stochastic TTS inferior because diversity and naturalness are different objectives. Full-context targets themselves have style variability; Gamma measures future-induced change in conditional means.

## T1.3 Proposition 2: bias correction with repeated recordings

$$
\begin{aligned}\bar Y_j&=n^{-1}\sum_{b=1}^nY_{jb},\quad \bar Y_\cdot=J^{-1}\sum_j\bar Y_j,\\ S_j&=(n-1)^{-1}\sum_b(Y_{jb}-\bar Y_j)(Y_{jb}-\bar Y_j)^{\mathsf T},\\\widehat\Gamma&=J^{-1}\sum_j\|\bar Y_j-\bar Y_\cdot\|^2-\frac{1-1/J}{Jn}\sum_j\operatorname{tr}(S_j).\end{aligned}
\tag{T1.5}
$$

**Symbols, dimensions, and conditions.** J is the number of equally weighted suffix classes; n≥2 is independent repeats per class. $Y_{jb}\in\mathbb R^q$ and $S_j:q\times q$. This formula targets a controlled uniform-J design; nonuniform suffixes need weighted variance and matching noise correction.

$$
\begin{aligned}\bar Y_j&=\mu_j+e_j,\quad\mathbb E e_j=0,\quad\operatorname{Cov}(e_j)=\Sigma_j/n,\\\mathbb E\left[J^{-1}\sum_j\|\bar Y_j-\bar Y_\cdot\|^2\right]&=J^{-1}\sum_j\|\mu_j-\bar\mu\|^2+\frac{1-1/J}{Jn}\sum_j\operatorname{tr}(\Sigma_j),\\\mathbb E\widehat\Gamma&=J^{-1}\sum_j\|\mu_j-\bar\mu\|^2.\end{aligned}
\tag{T1.6}
$$

**Symbols, dimensions, and conditions.** $\mu_j,e_j,\bar\mu\in\mathbb R^q$ and $\Sigma_j$ is the true within-class covariance. Independence gives Var(e_j−bar e)=Var(e_j)+Var(bar e)−2Cov(e_j,bar e); averaging over j yields the factor 1−1/J. Since S_j is unbiased for Sigma_j, subtraction removes the noise bias.

The estimate can be negative. Clamping to zero does not preserve unbiasedness or create a confidence lower bound. Use template-level bootstrap intervals, explicitly approximate rather than exact finite-sample guarantees. n=2 is noisy; use n=8/16 on a controlled validation subset. Shared TTS seeds induce correlation; paired-seed designs need a revised covariance correction.

## T1.4 Training, inference, experiments, and publication value

Freeze S6/VITS and the full-context teacher, then collect same-prefix/multiple-suffix Y. Train capacity-matched prefix-only and full-context regression heads using MSE and diagonal-Gaussian NLL comparisons. Compute T1.5 offline and fit an independent head to predict Gamma. Candidate suffixes must come only from the current LLM kernel. Online Gamma predictions remain estimates, not known irreducible error.

Validate three levels: a synthetic two-suffix experiment matching T1.4; semantic-focus data where the full-context head should outperform the prefix head; and natural incremental text where estimated Gamma should correlate with actual full-context revision. Plot empirical optimal/fixed/predicted-wait distortion frontiers for each latency budget. Without a coding-rate variable, do not call the curve a Shannon rate-distortion function.

**Novelty and gaps.** The mathematics behind T1.3/T1.6 is classical. A publishable contribution would require measurable causal TTS targets, reliable teacher-noise correction, new tightness/delay-frontier evidence, or stronger bounds for restricted generator classes. Relabeling a bound with VITS is insufficient. References: [R01,R21,R40]. Proofs are provided, not claims of first publication or human-perception guarantees.

# T2. Joint error bounds in acoustic rather than lexical geometry

## T2.1 Research goal and probability objects

The hypothesis is that not all future-word errors matter: only those changing current speech should increase waiting cost. For fixed H, let P be the true future distribution, Q the forecast distribution, K_w the true future-conditioned prosody kernel, and Khat_w the learned kernel. All Y lie in one bounded normalized prosody space, not unequal-length waveform spaces. P must be an explicitly defined text-arrival process, not automatically Qwen softmax.

```text
Qwen future distribution Q -> future-conditioned S6/explicit prosody kernel Khat
                              -> marginal acoustic belief Q Khat
independent full-context audit -> true/empirical future P and targets K
                              -> acoustic coupling cost / uncertainty certificate
risk-aware commitment -> VITS SDP + flow + waveform generation
```

## T2.2 Classical KL bound: proof and why it can be loose

$$
\begin{aligned}P_Y&=\sum_wP(w)K_w,\qquad Q_Y=\sum_wQ(w)\widehat K_w,\\\varepsilon_L&=\mathrm{KL}(P\Vert Q),\qquad\varepsilon_A=\mathbb E_{w\sim P}\mathrm{KL}(K_w\Vert\widehat K_w),\\\mathrm{KL}(P_{W,Y}\Vert Q_{W,Y})&=\varepsilon_L+\varepsilon_A.\end{aligned}
\tag{T2.1}
$$

**Symbols, dimensions, and conditions.** W is a discrete suffix/class and Y∈R^q. P,Q are mass functions; K are normalized densities or masses. Epsilon terms are dimensionless scalars in natural-log units. Require absolute continuity and finite KL. Split log(P(w)K_w(y)/(Q(w)Khat_w(y))) into two terms and integrate to obtain the chain rule.

$$
\begin{aligned}\mathrm{KL}(P_{W,Y}\Vert Q_{W,Y})&=\mathrm{KL}(P_Y\Vert Q_Y)+\mathbb E_{Y\sim P_Y}\mathrm{KL}(P(W\mid Y)\Vert Q(W\mid Y)),\\\mathrm{KL}(P_Y\Vert Q_Y)&\le\varepsilon_L+\varepsilon_A,\\\operatorname{Regret}(\hat a)&\le2\operatorname{TV}(P_Y,Q_Y)\le\sqrt{2(\varepsilon_L+\varepsilon_A)}.\end{aligned}
\tag{T2.2}
$$

**Symbols, dimensions, and conditions.** Apply the chain rule in the opposite order for the first line; nonnegative KL yields the second. For action losses bounded in [0,1], each expected-loss error is at most TV. Insert true-optimal a* and model-optimal ahat and use model optimality to obtain 2TV, then apply Pinsker. Regret is a true-distribution one-step loss gap, not an arbitrary multistep policy guarantee.

This is not a new theorem. Its weakness is practical: lexical differences irrelevant to current prosody inflate epsilon_L, and support mismatch can make the bound infinite. The original VITS posterior/prior KL is not epsilon_A. The next proposition uses transport between acoustic kernels to seek a task-relevant bound.

## T2.3 A task-specific transport bound: definition and proof

$$
\begin{aligned}\delta_H(w,w')&=W_1(K_w,K_{w'}),\\\mathsf{OT}_{\delta_H}(P,Q)&=\inf_{\pi\in\Pi(P,Q)}\sum_{w,w'}\pi_{ww'}\delta_H(w,w'),\\W_1(PK,Q\widehat K)&\le\mathsf{OT}_{\delta_H}(P,Q)+\mathbb E_{w\sim Q}W_1(K_w,\widehat K_w).\end{aligned}
\tag{T2.3}
$$

**Symbols, dimensions, and conditions.** $W_1$ is first-order Wasserstein distance with ground metric d on Y; assume finite first moments. $\pi$ is a coupling with marginals P,Q, a J×J matrix for J classes. Delta is a pseudometric on suffixes: different words can have zero distance. The two terms measure acoustically relevant forecast error and acoustic-model error.

Proof step 1: choose any future coupling pi. For each pair (w,w′), couple K_w and K_w′ optimally or arbitrarily close to optimally. Sampling the pair and then (Y,Y′) gives marginals PK and QK with expected cost sum pi·delta, bounding W1(PK,QK). Take the infimum over pi. Step 2: choose the same w for K_w and Khat_w to obtain W1(QK,QKhat)≤E_Q W1(K_w,Khat_w). Step 3: add them using the W1 triangle inequality. Finite J avoids measurable-selection issues; general spaces require standard measurable-kernel and coupling-selection conditions.

$$
\begin{aligned}
&|\ell(a,y)-\ell(a,y')|\le L_\ell d(y,y')\\
&\quad\Longrightarrow\quad\operatorname{Regret}(\hat a)
\le2L_\ell\left[\mathsf{OT}_{\delta_H}(P,Q)+\mathbb E_QW_1(K_w,\widehat K_w)\right].
\end{aligned}
\tag{T2.4}
$$

**Symbols, dimensions, and conditions.** $L_\ell\ge0$ is a Lipschitz constant uniform across allowed actions. Under any coupling, expected-loss difference is at most L_ell times expected distance; optimize the coupling and use the same two-insertion argument as T2.2. Discontinuous threshold 0/1 losses do not satisfy this formula without smoothing or margin assumptions.

This is a task-specific candidate proposition proved here, but coupling and transport stability are established, with Chain Rule Optimal Transport a direct precedent [R25,R41]. It must not be advertised as a wholly new Wasserstein theorem. Potential contributions are learnable acoustic equivalence classes, computable finite-sample certificates, and effective guidance for waiting policies.

## T2.4 A computable finite-class certificate

To avoid unknown true-kernel costs, start with an auditable J-class centroid predictor Khat_j=delta_vj, where v_j∈R^q is a frozen prosody plan, not the full random VITS distribution. Fix bounded feature ranges and a metric with diameter Delta. Independently audit real class-j Y samples to estimate e_j=E d(Y,v_j). Estimate Q by mapping Qwen futures to a frozen partition; P comes from the controlled true arrival process or sufficiently many independent natural samples.

$$
C_{ij}=d(v_i,v_j),\quad e_j=\mathbb E_{Y\sim K_j}d(Y,v_j),\qquad W_1(PK,Q\widehat K)\le\sum_jP_je_j+\mathsf{OT}_C(P,Q).
\tag{T2.5}
$$

**Symbols, dimensions, and conditions.** $C:J\times J$ and $e,P,Q\in\mathbb R^J$; e_j has the same units as d. First transport PK to P Khat at cost at most sum P e, then optimally transport the centroids. This order replaces unknown kernel-to-kernel distances by computable C.

$$
\begin{aligned}
t_P&=\min\left(1,\frac J2\sqrt{\frac{\log(6J/\delta_{conf})}{2n_P}}\right),\\
t_Q&=\min\left(1,\frac J2\sqrt{\frac{\log(6J/\delta_{conf})}{2n_Q}}\right),\\
e_j^U&=\min\left(\Delta,\bar e_j+\Delta\sqrt{\frac{\log(3J/\delta_{conf})}{2n_j}}\right),\\
W_1(PK,Q\widehat K)&\le\sum_j\widehat P_je_j^U+\mathsf{OT}_C(\widehat P,\widehat Q)\\
&\qquad+2\Delta t_P+\Delta t_Q.
\end{aligned}
\tag{T2.6}
$$

**Symbols, dimensions, and conditions.** $\delta_{conf}\in(0,1)$ is certificate failure probability, not acoustic distance delta_H. n_P,n_Q are independent future-sample counts, n_j class-audit counts; P_hat,Q_hat are frequencies and bar e are sample means. The guarantee holds with probability at least 1−delta_conf under fixed v/C/partition, independent i.i.d. samples, and bounded d. For n_j=0, set eU=Delta rather than inventing a narrow interval.

Proof: apply two-sided Hoeffding to J frequency coordinates, assigning delta_conf/3 separately to P and Q, giving TV≤J/2 times the maximum coordinate error. Apply one-sided Hoeffding to J e_j terms with the remaining delta_conf/3. On their joint event, sum P e≤sum P_hat eU+Delta tP. Since C is a centroid metric of diameter at most Delta, perturbing either OT marginal changes cost by at most Delta times its TV. Thus OT(P,Q)≤OT(P_hat,Q_hat)+Delta(tP+tQ), producing T2.6. A nonmetric C requires a different stability argument.

The certificate can be loose: large J or few independent templates can make tP≈1. Start with controlled J=4/8 semantic classes. Natural prefixes often occur once, so do not claim enough samples for per-prefix guarantees. Bound shared prefix families/known processes explicitly, or label natural-data results empirical. Entropic Sinkhorn is not unregularized OT; solve a small-J linear program and report feasibility residuals.

## T2.5 Data, training, online inference, and experiments

Use full-context Y from training templates to define J acoustic equivalence classes through clustering or explicit rules; freeze the partition before calibration/test. Train a Qwen adapter for class probabilities and S6/prosody kernels for class-conditioned Y. Compare real suffixes, LLM samples, and random suffixes to separate semantic value from generic completion regularization. Estimate eU on an independent audit set; never fit centroids on test data.

First complete the shared baseline and S6. Add a 2048→256→J class head and a (192+64+16)→128→q centroid head, proposing 30k updates. Choose class CE and Y Huber/MSE weights on validation. The bound is not itself a loss unless a differentiable OT surrogate is explicitly adopted and its bias reported. Online, true P is unavailable: estimate acoustic risk and calibrate with T3. Use the strict T2.6 certificate only in experimental settings with auditable P.

Experiments must vary acoustic suffix distance at fixed lexical KL; increase lexical mismatch within acoustic-equivalent classes; vary acoustic-kernel noise; sweep n/J and bound width; measure actual violations; and compare lexical entropy with acoustic-OT risk under matched latency. Negative criteria: bounds always wider than the loss range, or OT risk fails to beat a simple pause-boundary classifier. References: [R21,R25,R32,R34].

**Publication assessment.** This has a clearer theoretical question than rederiving an ELBO. The bounds above are proved under stated assumptions, but their broad mathematical novelty is unconfirmed. Competitive evidence still requires tighter useful certificates, nonvacuous real-data intervals, an effective algorithm, and speech benefits—not merely three classical inequalities.

# T3. Trajectory-calibrated irreversible speech commitment

## T3.1 Why per-prefix confidence is insufficient

A network’s SPEAK probability of 0.95 does not imply a 5% speech-error rate. Nor do separate 95% prefix sets imply 95% sentence-level safety. This proposal calibrates an entire text-arrival trajectory using its maximum prefix score, obtains simultaneous coverage, and then permits adaptive commitment. The mathematical foundation is split-conformal rank exchangeability; candidate task novelty lies in acoustic sets, whole-trajectory selection, and computable commitment certificates [R22–R24].

Assumptions are explicit: freeze the trained model; calibration/test trajectories are exchangeable under the declared distribution; incoming text is exogenous to playback actions; pending phone spans j and observable prefix-grid times k follow a predetermined rule. Y*_j comes from one full-context reference. The risk predictor uses prefix text, a fixed speaker/reference, and predetermined deterministic state, not generated audio that changes the coverage target. Feedback dialogues, distribution-shifted speakers, or unbounded trajectories require a new guarantee.

```text
TRAIN: prefix features -> mean mu[j,k,q] and positive scale s[j,k,q]
CALIBRATE: whole independent trajectories -> maximum normalized residual S_i
           order statistic -> one trajectory-level radius q_conf
ONLINE: observed prefix -> prosody set C[j,k]
        proposed plan or privately synthesized audio -> worst-case set distance
        certified distance <= tolerance -> SPEAK; otherwise WAIT
        deadline override -> flagged uncertified output, never counted as certified
```

## T3.2 Set construction, dimensions, and training

$$
\begin{aligned}\mu_{jk},s_{jk}&=F_\eta(H_k,c_j),\quad s_{jk,\ell}=\operatorname{softplus}(a_{jk,\ell})+s_{min},\\\mathcal L_{pred}&=\frac12\sum_{j,k,\ell\;valid}\left[\frac{(Y^*_{j,\ell}-\mu_{jk,\ell})^2}{s_{jk,\ell}^2}+2\log s_{jk,\ell}\right].\end{aligned}
\tag{T3.1}
$$

**Symbols, dimensions, and conditions.** $j=1…J_{span}$ indexes pending spans, $k=1…K_{obs}$ observable prefix positions, and $\ell=1…q$ prosody coordinates; J_span is not T2’s class count. Mu/s have conceptual shape $B\times J_{span}\times K_{obs}\times q$, stored sparsely over valid pairs. Proposed s_min=0.05 in standardized units. The constant qlog(2pi)/2 may be omitted in optimization. Conformal validity does not require the Gaussian predictor to be correct.

$$
S_i=\max_{(j,k,\ell)\in\mathcal G_i}\frac{|Y^*_{ij,\ell}-\mu_{ijk,\ell}|}{s_{ijk,\ell}},\quad b=\left\lceil(n_{cal}+1)(1-a_{risk})\right\rceil,\quad Q_{conf}=S_{(b)}.
\tag{T3.2}
$$

**Symbols, dimensions, and conditions.** $\mathcal G_i$ is the predetermined valid-coordinate set of trajectory i. S_i is a nonnegative scalar and n_cal counts independent calibration trajectories, not prefixes. $S_{(b)}$ is the b-th ascending score; set it to infinity if b>n_cal. a_risk∈(0,1) is the whole-trajectory miscoverage level.

$$
\begin{aligned}
\mathcal C_{jk}(H_k)&=\{r\in\mathbb R^q:|r_\ell-\mu_{jk,\ell}|\le Q_{conf}s_{jk,\ell}\ \forall\ell\},\\
P\big(\forall(j,k)&\in\mathcal G_{new}:Y^*_j\in\mathcal C_{jk}\big)\ge1-a_{risk}.
\end{aligned}
\tag{T3.3}
$$

**Symbols, dimensions, and conditions.** C_jk is a q-dimensional axis-aligned box, not a pointwise confidence probability. The probability is over the new trajectory and calibration sample, conditional on the frozen training model. Discrete pause coordinates may be embedded in bounded intervals, but an interval is not an exact binary-label probability.

Proof: exchangeability makes the rank of the test trajectory score symmetric among n_cal+1 scores. With continuous scores the rank is uniform; conservative quantiles handle ties. Thus P(S_new≤S_(b))≥b/(n_cal+1)≥1−a_risk. S_new≤Q_conf is equivalent to simultaneous inclusion of all valid (j,k,ell), establishing trajectory coverage. Correlated prefixes from one recording enter one maximum score, not independent calibration samples.

## T3.3 Computable commitment and post-selection guarantees

$$
\begin{aligned}d_w(u,v)&=\sum_{\ell=1}^q w_\ell|u_\ell-v_\ell|,\quad w_\ell\ge0,\quad\sum_\ell w_\ell=1,\\B_{jk}(\widehat r)&=\sum_\ell w_\ell\big(|\widehat r_\ell-\mu_{jk,\ell}|+Q_{conf}s_{jk,\ell}\big),\\\operatorname{SPEAK}_{jk}&\Longrightarrow B_{jk}(\widehat r)\le\epsilon.\end{aligned}
\tag{T3.4}
$$

**Symbols, dimensions, and conditions.** $u,v,\hat r\in\mathbb R^q$ and w is a fixed q-vector. Epsilon is a standardized prosody-distance tolerance. The farthest endpoint in each coordinate is abs(hat r−mu)+radius; because the box is a Cartesian product, B is the exact supremum of weighted L1 distance over it. Clipping the box to physically allowed ranges leaves B a conservative upper bound.

$$
P\left(\exists\text{ certified commit }(j,k):d_w(\widehat r_{jk},Y^*_j)>\epsilon\right)\le a_{risk}.
\tag{T3.5}
$$

**Symbols, dimensions, and conditions.** Commit times may depend adaptively on prefixes, buffers, predictions, and additional randomness, provided they select valid grid pairs and satisfy T3.4. On the simultaneous coverage event, every selected target is in its set and has distance≤B≤epsilon. Any violating certified commitment therefore implies coverage failure. No sum of pointwise error probabilities is required.

$$
P\left(\exists\text{ certified commit}:d_w(\Phi(\widehat y),Y^*)>\epsilon+\delta_{exec}\right)\le a_{risk}+b_{exec}.
\tag{T3.6}
$$

**Symbols, dimensions, and conditions.** $\Phi$ is a fixed waveform-to-prosody extractor with output in R^q. If the probability of any trajectory-level execution error d_w(Phi(yhat),rhat)>delta_exec is at most b_exec, the triangle inequality and union bound give this result. This must be a whole-trajectory execution guarantee, not an average-frame error or single-step beta.

A practical alternative privately synthesizes a candidate, extracts its actual prosody with Phi as hat r, and checks T3.4 before playback. This removes model plan-to-execution error but adds candidate synthesis, feature extraction, and boundary-stability costs; Phi’s measurement error still needs auditing. Either version controls violations relative to the selected reference and metric, not MOS or universal listener satisfaction.

T3.5 bounds the joint event of at least one certified error. It is not P(error|SPEAK)≤a_risk or a per-prefix conditional guarantee. Deadline-forced output failing the certificate is an uncertified override and is outside the guarantee. A system that almost always waits for the full sentence may be low risk but not usefully streaming; report certification rate, waiting distribution, and full-sentence fallback rate.

## T3.4 Calibration data, training, and evaluation

Use training data only for F_eta. Choose architecture, scale floor, w, epsilon, grid, and maximum length on validation; freeze them before a one-time calibration of Q_conf. CF-Prefix derivatives share templates, suffixes, speakers, and repeats; take a template-level maximum score over related trajectories. Then n_cal=80, not 80×4×3×2. With 80 independent groups, a_risk=0.01 gives an infinite quantile; more calibration groups are needed. Splitting prefixes does not create independent samples.

Implementation: a two-layer MLP or two-layer causal Transformer, width 256, output 2q, trained for a proposed 30k updates. Inputs are frozen current Qwen states, observed phones, and speaker condition. No full-sentence punctuation, future G2P, or teacher states are allowed. Assess scale calibration on held-out data without changing Q_conf using test outcomes. Cache mu/s per (j,k); private candidates may use S1, S3, or S6 backends.

Report trajectory coverage with exact binomial/Clopper–Pearson intervals, certified and overall violation rates, override share, subgroup coverage by language/length/speaker, average set width, p50/p95 first-playable and waiting delays, and speech/content metrics. Compare pointwise conformal, union-bound allocation, fixed lookahead, entropy gating, and X2Streaming-style mechanisms using the same backend. Treat distribution shift as a separate stress test; do not claim assumption-free validity.

**Novelty boundary and readiness.** Maximum-score conformal is a classical construction; no claim of first appearance in general prediction is made. A competitive contribution would be whole-trajectory modeling for irreversible speech, private-candidate acoustic certificates, an efficient implementation, and a nontrivial coverage–latency frontier. Huge sets and full-sentence waiting are not useful evidence. T2 acoustic classes may tighten sets, but every changed predictor requires recalibration.

# T4. Prefix consistency, tractable density, and bounded-memory causal flows

## T4.1 Scope, modules, and noncausal risks

Invertibility does not imply streaming. Replace the acoustic-flow conditioner with an inter-block causal, intra-block parallel network while preserving tractable coupling Jacobians. The training posterior may see full audio; the generative prior may not. Autoregressive/causal flows are established [R20]; the candidate contribution is VITS-specific clock alignment, cache-truncation control, and commitment consistency—not the triangular Jacobian itself.

```text
available text c_k + future belief r_k -> block condition
committed latent state s_{k-1} + fresh noise eps_k -> invertible block g -> z_k
z_k + c_k -> deterministic state update -> s_k
z_k + committed waveform cache -> causal waveform block
training full-audio posterior -> sequence KL against the same causal prior
```

Propose blocks of 8/16/32 high-rate latent frames, about 85/171/341 ms of acoustic content—not total latency. Start with four affine coupling blocks split 96/96 channels; use a three-layer width-192 causal Conv1d conditioner with kernel 3 and a 64-dimensional projected state broadcast over the block. These are new experimental settings. Validate fixed lengths before variable phone blocks instead of changing multiple clocks simultaneously.

## T4.2 Causal kernels and prefix-invariance proof

$$
\epsilon_k\sim\mathcal N(0,I_{D_k}),\quad z_k=g_{\theta,k}(\epsilon_k;s_{k-1},c_k),\quad s_k=F_\theta(s_{k-1},z_k,c_k),\quad D_k=192T_k.
\tag{T4.1}
$$

**Symbols, dimensions, and conditions.** Flattened z_k,epsilon_k have dimension D_k, otherwise 192×T_k. $s_k\in\mathbb R^{C_s}$ with proposed C_s=64; c_k is available text/prosody conditioning. Noise is independent across blocks, g is invertible in epsilon for fixed conditions, and F is deterministic. Distinguish this flow-state width from S3 speaker width in code as flow_state_dim versus speaker_dim.

$$
p_\theta(z_{1:K}\,\Vert\,c_{1:K})=\prod_{k=1}^Kp_\theta(z_k\mid z_{<k},c_{\le k}),\qquad \int p_\theta(z_{1:K}\Vert c_{1:K})\,dz_{k+1:K}=p_\theta(z_{1:k}\Vert c_{1:k}).
\tag{T4.2}
$$

**Symbols, dimensions, and conditions.** The double bar denotes causally conditioned kernels under a prescribed exogenous input path, not arbitrary observational conditioning on future feedback. Integrate normalized factors backward from the last; each gives one, leaving the prefix product. With fixed history, seeds, and block rules, appending input does not alter committed latent distributions or samples.

Do not assert unconditional observational equality p(z_past|H_all)=p(z_past|H_past): future text generated from past-audio feedback can change the posterior of the past. A nonanticipatory generation property may hold without that observational independence. Prefix consistency under fixed chunk rules also does not imply identical waveforms under arbitrary chunk sizes; that needs additional state/position design.

## T4.3 Invertible coupling and exact density

$$
\begin{aligned}z_a&=\epsilon_a,\quad (a,b)=C_\theta(\epsilon_a,s_{k-1},c_k),\\z_b&=\epsilon_b\odot\exp(a)+b,\qquad\epsilon_b=(z_b-b)\odot\exp(-a),\\\log|\det J_g|&=\sum_{t,c}a_{ct},\qquad\log p(z_k\mid s,c)=\log p_0(\epsilon_k)-\sum_{t,c}a_{ct}.\end{aligned}
\tag{T4.3}
$$

**Symbols, dimensions, and conditions.** $\epsilon_a,\epsilon_b,z_a,z_b,a,b\in\mathbb R^{96\times T_k}$. Sum over transformed half-channels and valid frames only. C does not depend on epsilon_b, making the split Jacobian triangular. Proposed a=0.5tanh(a_raw) avoids numerical overflow. Channel flips/mixing across layers preserve invertibility; sum their log determinants.

$$
\begin{aligned}\log p(z_{1:K}\Vert c_{1:K})&=\sum_k\left[\log p_0(\epsilon_k)+\log\left|\det\frac{\partial\epsilon_k}{\partial z_k}\right|\right],\\\mathcal E_{causal}&=\sum_k\mathbb E_q\log p_\gamma(m_k\mid z_{\le k},c_{\le k})\\&\quad-\sum_k\mathbb E_{q(z_{<k})}\mathrm{KL}\left(q_\phi(z_k\mid z_{<k},x_{1:K},c_{1:K})\Vert p_\theta(z_k\mid z_{<k},c_{\le k})\right).\end{aligned}
\tag{T4.4}
$$

**Symbols, dimensions, and conditions.** The global Jacobian is block lower triangular, so its determinant factors into diagonal-block determinants, giving the first equation. For the second, chain-factor q, apply multiply/divide/Jensen to log∫p, and group blockwise logq−logp. Use nonoverlapping valid Mel coordinates; avoid double-counting overlapping windows. Duration/future-r conditions are omitted for readability and add terms as in G5/S6.5.

## T4.4 Bounded-memory error: open-loop versus closed-loop

$$
\|F(s,z,c)-F(s',z,c)\|\le\kappa\|s-s'\|,\quad\kappa<1\ \Longrightarrow\ \|s_k-s'_k\|\le\kappa^L\|s_{k-L}-s'_{k-L}\|.
\tag{T4.5}
$$

**Symbols, dimensions, and conditions.** This compares states receiving identical z/c inputs for the final L steps. Repeated application of the Lipschitz inequality gives the bound. It applies when truncated history is replayed with identical cached inputs, not automatically to two systems generating different z online.

$$
\begin{aligned}\|g(\epsilon;s,c)-g(\epsilon;s',c)\|&\le L_g\|s-s'\|,\\\|F(s,z,c)-F(s',z',c)\|&\le L_s\|s-s'\|+L_z\|z-z'\|,\\\kappa_{cl}&=L_s+L_zL_g<1\ \Longrightarrow\ \|s_k-s'_k\|\le\kappa_{cl}^{L}\|s_{k-L}-s'_{k-L}\|.\end{aligned}
\tag{T4.6}
$$

**Symbols, dimensions, and conditions.** Couple two generated trajectories using identical epsilon and c. L_g,L_s,L_z are uniform nonnegative Lipschitz constants. Bound z discrepancy with the first equation, substitute into the second to obtain kappa_cl, and iterate L steps. Bounding a state matrix’s spectral norm alone does not ensure kappa_cl<1; g’s scale and conditioner also require auditing.

$$
\|\widehat y_k-\widehat y'_k\|\le(L_{yz}L_g+L_{ys})\,\kappa_{cl}^{L-1}\Delta_s.
\tag{T4.7}
$$

**Symbols, dimensions, and conditions.** Assume uniform decoder Lipschitz constants L_yz and L_ys for current z and its cached-state projection, with cache error included in s; Delta_s bounds the initial truncation discrepancy. Bound z error by L_g times state error, then apply the triangle inequality. Ordinary HiFi-GAN does not automatically have known tight constants; empirical perturbation sensitivity is not a global proof.

For a rigorous contraction experiment, use mean-only coupling (a=0), fixed base variance, and spectral control of all conditioning networks. If affine scale depends on state, unbounded Gaussian noise can destroy a global state-Lipschitz constant because the derivative contains epsilon_b. Prove a bounded-noise-event result with an additional tail failure probability, or use the additive structure; bounding a with tanh alone does not establish finite global L_g.

## T4.5 Training/inference implementation and test matrix

Use long within-utterance spans and genuinely continuous paragraphs from P1, preserving phone boundaries and absolute latent-frame indices. Replace only flow first, freeze posterior/decoder for 50k updates, then train a causal/limited-right-context decoder for 50k–100k and jointly tune for 20k. Randomize block length under fixed positional rules. Explicitly cache convolution context, transposed-convolution upsampling phase, and uncommitted tails. Use absolute-frame counter-based randomness, not the same restarted seed for each block.

Required tests: double-precision low-dimensional inverse residual<1e-8; analytic versus autograd logdet error<1e-6; future-text changes cannot alter committed samples; offline/streaming parity under identical block plans within declared floating-point tolerance; separately quantify changed block plans rather than claiming exact equality; compare L=1/2/4/8 truncation errors with computable bounds. These are test-design thresholds, not upstream-model guarantees.

**Publication assessment.** T4.2–T4.6 use established kernel normalization, triangular Jacobians, and Lipschitz recursion. A competitive result would simultaneously retain tractable density, within-block parallelism, and useful cache bounds without destroying quality. Enormous global constants or severe expressive degradation make a correct theorem unhelpful. This can form an independent generative-model project with S3’s dual-clock AudioVAE.

# T5. Optimal commitment under compute and risk constraints

## T5.1 Motivation, state, and physical time

Fixed lookahead and simple information thresholds ignore audio buffers, token arrival rates, candidate-computation cost, and repeated waiting. Treat commitment as finite-horizon stopping: speak now or pay physical time for genuinely arriving text. Dynamic programming and approximate optimal stopping are established [R26]; the candidate contribution combines S6 acoustic risk, T3 certificates, and inference budgets, not a new Bellman equation.

```text
actual arrival trace + buffer + compute ledger -> observable state x_k
S6/prosody predictor -> STOP risk R_k
Qwen predictive kernel + arrival model -> WAIT continuation estimate
T3 certificate -> admissible certified STOP mask
finite-horizon value head -> choose action
STOP -> waveform already synthesized or render with counted cost -> playback
WAIT -> advance real clock and observe actual text, never the sampled fake suffix
```

State x_k includes visible text/cache features, pending phones, buffered audio seconds b_k, elapsed wait t_k, and remaining compute c_k. Use seconds consistently; compute may be measured in GPU-ms or vector-field calls but must map to physical latency. Token arrivals need not have fixed delays; use recorded traces or an explicit conditional arrival model. Proposed test caps of K_max=32 decisions or 400 ms are not performance guarantees.

## T5.2 Deriving the Bellman equation

$$
\begin{aligned}R_k(x)&=\inf_{a\in\mathcal A_{speech}(x)}\mathbb E[\ell(a,Y^*)\mid x_k=x],\\ V_K^*(x)&=R_K(x),\\ Q_k^*(x,\mathrm{STOP})&=R_k(x),\\ Q_k^*(x,\mathrm{WAIT})&=\mathbb E[c_k(x,\Delta t,x')+V_{k+1}^*(x')\mid x],\\ V_k^*(x)&=\min\{Q_k^*(x,\mathrm{STOP}),Q_k^*(x,\mathrm{WAIT})\}.\end{aligned}
\tag{T5.1}
$$

**Symbols, dimensions, and conditions.**  $\mathcal A_{speech}$ is the set of speech plans; a is not the WAIT/STOP action. Ell is normalized prosody loss; R,V,Q are scalar. $c_k$ is nonnegative physical waiting/computation/buffer-underflow cost, for example lambda_t Delta t + lambda_u underflow, with lambda_t in loss/second. Define a terminal STOP rule or infeasibility penalty explicitly.

Proof: at the terminal step, no further observation is available, so optimal remaining loss is R_K. Induct backward: a policy first either stops or waits. Stopping costs R_k; waiting incurs c and reaches x′, whose optimal continuation cost is V*_(k+1) by induction. Conditional expectation under the true transition and minimization give the last equation. If x is not Markov-sufficient, use the full observed history as state; the equation remains valid but approximation is harder.

$$
R_k(x)-\mathbb E[R_{k+1}(x')\mid x]>\mathbb E[c_k\mid x]
\tag{T5.2}
$$

**Symbols, dimensions, and conditions.** This is the wait condition only when one extra observation must be followed by stopping. It is not the general T5.1 optimum because V*_(k+1) can be below R_(k+1). Under squared loss R is conditional covariance trace; under log scoring it is conditional entropy. Do not interchange the objectives.

## T5.3 Proposition: commitment regret from approximate values

$$
\begin{aligned}
\sup_{x,a}|\widehat Q_k(x,a)-Q_k^*(x,a)|&\le\varepsilon_k,\\
\widehat\pi_k(x)&\in\arg\min_a\widehat Q_k(x,a)\\
\Longrightarrow\quad J(\widehat\pi)-J(\pi^*)&\le2\mathbb E_{\widehat\pi}\sum_{k\le\tau}\varepsilon_k.
\end{aligned}
\tag{T5.3}
$$

**Symbols, dimensions, and conditions.**  $a\in\{STOP,WAIT\}$; $\varepsilon_k$ is a uniform error bound over relevant state/action pairs, tau the stopping time, and J the true expected total loss including latency. This is stronger than average validation MSE and does not follow automatically from training error. A finite K keeps the sum finite.

$$
\begin{aligned}Q_k^*(x,\widehat a)-V_k^*(x)&=Q_k^*(x,\widehat a)-Q_k^*(x,a^*)\\&\le[Q_k^*(x,\widehat a)-\widehat Q_k(x,\widehat a)]+[\widehat Q_k(x,a^*)-Q_k^*(x,a^*)]\\&\le2\varepsilon_k.\end{aligned}
\tag{T5.4}
$$

**Symbols, dimensions, and conditions.** The omitted term Qhat(x,ahat)−Qhat(x,a*) is nonpositive because ahat minimizes Qhat. Summing true stage cost plus next V* minus current V* causes intermediate value terms to telescope in expectation. The terminal value contains STOP risk, leaving J(hatpi)−V0*, which proves T5.3.

$$
|\widehat{\mathbb E}[c+V']-\mathbb E[c+V']|\le e_c+(K-k)C_{max}\,\operatorname{TV}(\widehat P_{next},P_{next})+e_{V'}.
\tag{T5.5}
$$

**Symbols, dimensions, and conditions.** Assume c∈[0,C_max] and remaining value∈[0,(K-k)C_max], including terminal loss in an appropriate stage bound. e_c is immediate expected-cost error and e_V′ uniform continuation-value error. Bound the integral discrepancy by function range times TV and apply the triangle inequality. This shows amplification from LLM/arrival-model mismatch; it does not establish reliable estimation from a few natural examples.

## T5.4 Extra LLM rollouts are not actual future observations

$$
\widetilde W\perp W^*\mid H\quad\Longrightarrow\quad I(W^*;\widetilde W\mid H)=0.
\tag{T5.6}
$$

**Symbols, dimensions, and conditions.**  $\tilde W$ is a model suffix drawn with an independent seed and $W^*$ is the actually arriving suffix. State this conditional-independence assumption given model/prefix. Independence makes the joint equal the product of conditionals, so the mutual-information log ratio is zero. Shared random state or coupled decoding requires a different model.

Thus a THINK/rollout action can reduce finite-Monte-Carlo estimation error or improve a resource-limited decision maker; it is not the Bayes information gained by observing an actual new token. A metareasoning state can include sample means, variances, and counts, paying real compute time before sampling again. The first implementation need not use complex THINK reinforcement learning: compare fixed M=1/2/4/8 with all computation included in latency.

## T5.5 A restricted but rigorous earliest-certified-release property

$$
\tau_0=0,\qquad\tau_i=\min\{k\ge\tau_{i-1}:\mathrm{Cert}_i(k)=1\}.
\tag{T5.7}
$$

**Symbols, dimensions, and conditions.**  i indexes ordered speech blocks; Cert_i(k) is a certification indicator from exogenous text and fixed candidate rendering. Multiple blocks may be released at one instant. Assume certificates and future candidates do not depend on past release actions, there are no additional capacity conflicts, and the objective is a nonnegative weighted sum of release delays. If infeasible, tau_i=infinity.

Proposition: within this restricted class, every tau_i is no later than that of any order-preserving, certified-only policy sigma_i. Induction: tau_0=sigma_0. If tau_(i−1)≤sigma_(i−1), then sigma_i≥sigma_(i−1)≥tau_(i−1) and Cert_i(sigma_i)=1, so sigma_i is feasible in T5.7 and tau_i≤sigma_i. This gives pathwise minimization of any nonnegative weighted delay sum. Actual S6 feedback can violate exogenous-candidate assumptions; the result does not extend automatically to all synthesis policies.

## T5.6 Data, value training, deployment, and evaluation

Start with a controlled finite tree: J=2/4 suffix classes, K≤12 observable prefixes, known arrival probabilities/times, and a fixed low-dimensional target generator. Exact dynamic programming produces Q* to test T5.3 and one-step thresholds. Then build rollout data from CF-Prefix suffixes, recorded arrival times, and full-context targets. A single natural suffix does not reveal every branch’s Q*; teacher sampling is an approximation whose variance must be reported.

Train a two-layer MLP with a 256-dimensional pooled-Qwen projection, r64, rho16, three buffer/wait/compute scalars, and four risk statistics: 343 inputs and two STOP/WAIT Q outputs. Freeze TTS and train 30k updates on offline DP/rollout targets; then aggregate states visited by the current policy in training-distribution arrival simulations, never test. T3 supplies an action mask: disallow certified STOP when uncertified; at deadlines report infeasibility or an explicit override.

Online, update at actual token arrivals or timer ticks rather than busy-looping without information. Rendering may overlap upstream LLM compute, but measure total delay from actual timestamps. Evaluate true objective, latency Pareto frontier, violations, buffer underflow, call counts, and regret under mismatch. Controls: fixed waits, one-step VOI, compute-unaccounted policy, oracle DP, unconstrained DP, and T3 earliest-certified release.

**Publication boundary.** Bellman recursion, 2epsilon regret from uniform Q error, and earliest-release induction are not wholly new theories. Potential distinctiveness lies in combining finite-computation error, acoustic-impact geometry, and irreversible-speech certificates into an implementable policy with nontrivial tightness/lower-bound evidence. A toy-tree proof without real speech latency–quality gains does not support a strong application claim.

# 4. Recommended combinations, experiment matrix, and failure criteria

## 4.1 Primary route: S6 + T2 + T3, with T5 as an extension

Focus a first paper on one question: with the same streaming VITS backend and latency budget, is the acoustic impact of future language more useful than token entropy for irreversible commitment? S6 supplies stateful probabilistic durations, T2 task-relevant error geometry, and T3 trajectory-level risk semantics. Keep S1 diffusion, S3 AudioVAE, and S4 speech tokens out of the minimal system so attribution remains identifiable.

```text
data/manifest + splits + arrival traces
   -> frozen Qwen prefix extractor (2048)
   -> phone adapter (192) + future head (64)
   -> S6 rhythm state (16) + original VITS SDP
   -> T2 acoustic class/prosody probe
   -> T3 calibrated set + candidate waveform validation
   -> certified commit or explicit fallback
   -> conditional VITS prior/flow + stateful waveform decoder
TRAIN ONLY: full-context prosody posterior, VITS posterior, MAS, GAN discriminators
OPTIONAL TEACHER: frozen CosyVoice -> synthetic counterfactual targets, never online
```

| Run | Only changed factor | Question to answer |
|---|---|---|
| E0 | Offline VITS | Is the full-text reference/baseline reliable? |
| E1 | Causal chunks, fixed wait | What is lost purely from streaming? |
| E2 | Qwen hidden adapter | Are improvements just semantic features? |
| E3 | Future r, no state | Does future prediction change speech appropriately? |
| E4 | Rhythm rho state | Does history add independent benefit? |
| E5 | Same backend, token-entropy policy | Lexical-uncertainty baseline |
| E6 | Same backend, acoustic-impact policy | Does task geometry beat entropy? |
| E7 | Trajectory calibration | Is coverage useful and reproducible? |
| E8 | Compute-accounted stopping | Does added decision logic yield net benefit? |

Use identical data, LLM checkpoint, G2P, acoustic backend, and arrival traces across E runs. Report at least three seeds; choose models on validation only, then calibrate E7 after freezing. Log teacher-synthesis and LLM-rollout time and cost/energy where available rather than hiding offline data expense. If small fixed lookahead already nearly reaches the full-context ceiling, avoid a complex controller chasing negligible differences.

## 4.2 Independent alternative: continuous latents with S3 + T4

This alternative studies low-rate AudioVAE alignment compatibility and strictly causal priors rather than only future-text decisions. Compare original VITS, the same AudioVAE with noncausal/causal priors, naive versus overlap adapters, and evaluable-density coupling versus nonlikelihood CFM. Report a multiobjective frontier of reconstruction, intelligibility, latent bandwidth, cache, and latency. Do not simultaneously replace many modules in the primary route and report one MOS.

## 4.3 Experiment artifacts and result templates

```text
experiment_id, git_commit, weights_sha256, data_manifest_sha256
seed, train_steps, seen_audio_hours, effective_batch, precision
trainable_params, frozen_params, GPU_model, peak_memory_bytes
RTF_p50, RTF_p95, first_playable_ms_p50, first_playable_ms_p95
wait_ms_p50, wait_ms_p95, underflow_events, override_fraction
WER, CER, deletion_rate, repetition_rate, speaker_similarity
prosody_distance, pause_error, trajectory_coverage, violation_fraction
human_AB_preference, confidence_interval, annotation_protocol
status: PLANNED | TRAINED | EVALUATED | AUDITED
# Use null for unmeasured metrics; never fill with guessed performance.
```

Go/no-go rules are internal decisions, not conference thresholds: every main claim has a direct control; probability and shape tests pass; the method improves on simple baselines at matched waiting budgets without degrading content; calibration is not achieved by infinite sets or full-sentence fallback; and assumptions are auditable on the test domain or claims are explicitly empirical. Diagnose failures rather than masking them with a larger LLM.

# 5. Mathematical audit, novelty verification, and submission readiness

## 5.1 Separate proved results, existing tools, and unfinished claims

The derivations are checkable under listed assumptions, but a proof does not establish first appearance in the literature. Further verification found Chain Rule Optimal Transport, which already studies mixture bounds using conditional-distribution distances as ground costs [R41]. T2.3’s basic structure is therefore established, not a wholly new transport theorem. Position contributions around acoustic error objects, computable finite-sample certificates, commitment algorithms, and nontrivial speech evidence.

Prosodic-boundary streaming post-training is recent direct related work [R42], and speaking-rate control also has an activation-steering route without retraining [R43]. S6 must compare lightweight control or boundary-classification baselines, not merely an old uncontrolled VITS. Theory novelty searches must include online prediction, selective risk, mixture transport, and causal generation—not only TTS.

| Proposal | Established foundation | Potential distinctive contribution |
|---|---|---|
| S1 | Diffusion/VAE bounds | Validated multimodal future prosody, not merely replacing a Gaussian |
| S2 | CFM/CNF/trace estimation | Compute-efficient streaming latent paths with useful error control |
| S3 | AudioVAE/conditional flows | Dual-clock compatibility and generatability/fidelity tradeoffs |
| S4 | Discrete–continuous ELBO | Nonredundant duration control and semantic/acoustic division |
| S5 | WavLM adversarial features | Usually an auxiliary improvement |
| S6 | State VAEs/stochastic durations | Testable separation of future context and rhythm state |
| T1 | Conditional expectation/ANOVA | Measurable causal prosody gaps and noise-corrected evaluation |
| T2 | KL/OT/CROT/Hoeffding | Task-specific, nonvacuous, auditable joint certificates |
| T3 | Trajectory max-score conformal | Irreversible-audio certificates with useful latency |
| T4 | Triangular flows/Lipschitz recursion | Joint density, prefix, and cache-error design |
| T5 | Bellman/approximate DP | Risk-constrained release with real compute costs |

## 5.2 Review criteria and honest current status

The official NeurIPS 2026 review guidance emphasizes technical quality, clarity, significance, and originality; theory contributions particularly need correct proofs and appropriate assumptions. Valuable work need not invent every mathematical tool [R38]. A detailed design nevertheless remains different from a completed paper. None of the six full TTS systems has been trained or listening-tested here, so none can be marked as already meeting an acceptance requirement.

A reasonable assessment is: S5 alone is weak; S1/S2/S4 must outperform straightforward existing combinations; S3/T4 is generative-model oriented and engineering-heavy; S6/T2/T3 best concentrates on one speech-and-theory question. Competitive submission still needs independent novelty verification, strong baselines, a substantive reproducible benefit, agreement between assumptions and deployment, and public or auditable reproducibility material. No title, equation count, or single metric guarantees acceptance.

## 5.3 Checks that prevent invalid derivations

Check support and normalization; distinguish continuous-Dirac KL from discrete-tokenizer point masses; give each duration its corresponding latent dimension; do not identify GAN loss with observation likelihood, CFM MSE with log density, or a log mixture with a mean log without explanation. Check signs when replacing logp by a lower bound. Record stochastic-trace and numerical-ODE error.

Check the unit of probability guarantees: frame, prefix, utterance, or independent template. Audit margins, bounded features, and Lipschitz assumptions; match future probabilities to deployment; exclude full-text frontend leakage; never rerender committed audio. A mean-error bound is not a human-naturalness bound. Every new latent state needs a normalized distribution, parameterization, training posterior, and inference sampling path.

# 6. Reproduction starter, executable scope, and delivery notes

## 6.1 What the starter includes and excludes

The companion ZIP provides standalone CPU-runnable modules and numerical experiments: conditional affine-coupling inverse/Jacobian checks, state/Gaussian-head shape checks, T1 noise correction, T2 discrete transport bounds, T3 trajectory-coverage simulation, T5 finite-tree stopping, and a Gaussian prosody trainer reading NPZ features. It contains no restricted speech data, model weights, complete VITS fork, or six trained systems.

```bash
python -m venv .venv
# Activate this environment using the command for your operating system.
python -m pip install -r requirements.txt
python tests.py
python theory_lab.py --out results/theory.json
python train_prosody.py --demo --out results/prosody_demo
# Real features, after executing the extraction protocol in Chapters 1-3:
python train_prosody.py --train features_train.npz --valid features_valid.npz --out results/prosody_real
python validate_manifest.py data/manifest.jsonl
```

The NPZ trainer interface is context float32[n,context_dim], target float32[n,q], optional mask bool[n,q], and optional but recommended group_id strings[n]. Export context from frozen Qwen/phone adaptation; extract targets with D1 and normalize using training statistics. The program rejects overlapping train/validation groups. It validates the probabilistic-head data path, not automatic text-to-speech synthesis.

## 6.2 From starter to full speech experiments

Gate 1: obtain authorized data and chosen official checkpoints, pin commits/hashes, and reproduce unmodified VITS reconstruction/inference. Gate 2: extract manifest-linked features, run the validator and probability head, and prove prefix inputs contain no future leakage. Gate 3: implement one component interface, first under teacher conditions and then prior conditions, passing shape/gradient tests. Gate 4: add streaming caches and actual arrival times. Gate 5: freeze, calibrate, then run test and human evaluation.

Separate data, models, losses, streaming, evaluation, and theory directories. Give each proposal its own config rather than one huge if/else mixing probabilistic objectives. The first integrated VITS train_step should log scalar losses, lengths, mask coverage, gradient norms, and posterior/prior reconstruction examples. Update locks and regression tests for every model-interface or library-version change.

## 6.3 Scope of actual executed results

Local starter tests are saved separately as results/local_validation.json and a test log. They validate only the stated code paths and synthetic-data properties—not speech quality, global theorem novelty, or conference acceptance. PDF math/text and Markdown are generated from the same source with matching references. Treat this handbook as an auditable specification, not a fully validated final algorithm.

# Appendix A. Expanded variational algebra and dimension checks

## A.1 Diagonal-Gaussian KL without skipped algebra

$$
\begin{aligned}\log q(x)&=-\frac D2\log(2\pi)-\sum_j\log s_{q,j}-\frac12\sum_j\frac{(x_j-\mu_{q,j})^2}{s_{q,j}^2},\\\log p(x)&=-\frac D2\log(2\pi)-\sum_j\log s_{p,j}-\frac12\sum_j\frac{(x_j-\mu_{p,j})^2}{s_{p,j}^2},\\E_q[(x_j-\mu_{p,j})^2]&=E_q[(x_j-\mu_{q,j})^2]+(\mu_{q,j}-\mu_{p,j})^2,\\E_q[\log q-\log p]&=\sum_j\log\frac{s_{p,j}}{s_{q,j}}+\frac12\sum_j\left[\frac{s_{q,j}^2+(\mu_{q,j}-\mu_{p,j})^2}{s_{p,j}^2}-1\right].\end{aligned}
\tag{A1}
$$

**Symbols, dimensions, and conditions.** $x,\mu_q,\mu_p,s_q,s_p\in\mathbb R^D$, with positive scales and $j=1,\ldots,D$. Expanding the third line eliminates its cross term because $E_q(x_j-\mu_{q,j})=0$. The expected self-standardized square is one. Substituting both expectations in the difference of the first two lines yields G8. Sum only over valid, nonpadding dimensions.

## A.2 Completing the square for the diffusion posterior

In S1, $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_{v=1}^t\alpha_v$ are scalars. Put $u=r_{t-1}\in\mathbb R^R$, $R=64$. For $t\ge2$, the posterior is proportional to the product of two known Gaussians. Collect terms independent of u into a constant and compute precision and mean explicitly.

$$
\begin{aligned}q(u\mid r_t,r_0)&\propto q(r_t\mid u)q(u\mid r_0),\\-2\log q(u\mid r_t,r_0)&=\frac{\|r_t-\sqrt{\alpha_t}u\|^2}{\beta_t}+\frac{\|u-\sqrt{\bar\alpha_{t-1}}r_0\|^2}{1-\bar\alpha_{t-1}}+C,\\\Lambda_t&=\left(\frac{\alpha_t}{\beta_t}+\frac1{1-\bar\alpha_{t-1}}\right)I_R=\frac{1-\bar\alpha_t}{\beta_t(1-\bar\alpha_{t-1})}I_R,\\b_t&=\frac{\sqrt{\alpha_t}}{\beta_t}r_t+\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}r_0,\\\widetilde\beta_t I_R&=\Lambda_t^{-1},\qquad\widetilde\mu_t=\Lambda_t^{-1}b_t.\end{aligned}
\tag{A2}
$$

**Symbols, dimensions, and conditions.** $\Lambda_t\in\mathbb R^{R\times R}$ is precision, $b_t,\widetilde\mu_t\in\mathbb R^R$, $\widetilde\beta_t>0$ is scalar variance, and C is independent of u. Completing the square uses $u^T\Lambda_tu-2b_t^Tu=(u-\Lambda_t^{-1}b_t)^T\Lambda_t(u-\Lambda_t^{-1}b_t)-b_t^T\Lambda_t^{-1}b_t$. This yields both coefficients of S1’s posterior mean. Treat t=1 through the reconstruction term instead of dividing by zero.

$$
\begin{aligned}\mathcal B_D(r_0,c)=&\ E_q\log p_\psi(r_0\mid r_1,c)-\mathrm{KL}(q(r_D\mid r_0)\Vert p(r_D))\\&-\sum_{t=2}^D E_{q(r_t\mid r_0)}\mathrm{KL}(q(r_{t-1}\mid r_t,r_0)\Vert p_\psi(r_{t-1}\mid r_t,c)).\end{aligned}
\tag{A3}
$$

**Symbols, dimensions, and conditions.** D here counts diffusion steps, unlike the vector dimension in A1; $c\in\mathbb R^{256}$ and every KL is scalar. Substitute the reverse conditional identity $q(r_{1:D}|r_0)=q(r_D|r_0)\prod_{t=2}^Dq(r_{t-1}|r_t,r_0)$ into S1’s log-density ratio. The t=1 term is reconstruction and remaining ratios group into KL terms. Equivalence requires the corresponding weights, variances, and constants; arbitrary noise-MSE weighting is not automatically this bound.

## A.3 SDP transformations and inference length

$$
\begin{aligned}a&=d-u\in\mathbb R_+^N,\qquad b=\log a\in\mathbb R^N,\\e&=f_\alpha([b;v];c)\in\mathbb R^{2N},\\\log p_\alpha(a,v\mid c)&=\log p_0(e)+\log|\det J_{f_\alpha}([b;v])|-\sum_{i=1}^N\log a_i.\end{aligned}
\tag{A4}
$$

**Symbols, dimensions, and conditions.** d,u,v follow G9; $[b;v]\in\mathbb R^{2N}$ and $J_f\in\mathbb R^{2N\times2N}$. The total Jacobian is $J_f\operatorname{diag}(1/a_1,\ldots,1/a_N,1,\ldots,1)$, whose log determinant gives the second and third terms. The continuous cell $(d-1,d)$ defines p(d). At inference draw positive a and use the matching integer rule $d=\lceil a\rceil$. Temperature, length scaling, clipping, or minimum-duration changes alter the deployed distribution and require recalibration.

## A.4 Implementation audit of dimensions and units

Read symbols within the scope stated after each formula: D is flattened dimension in Gaussian/CNF formulas but diffusion-step count in S1/A3; d is an integer-duration vector while $d(\cdot,\cdot)$ is a specified metric; s may denote a probability scale, a speaker vector, or T4 cache state. In code use latent_dim, diffusion_steps, duration_frames, metric, speaker_embed, scale_std, and cache_state rather than passing ambiguous single letters across modules. Convert seconds, frames, and samples only through saved sample_rate/hop/latent_rate metadata.

# References and official implementation index

The following are primary paper pages, author/institution repositories, or official data pages. A 2026 technical report or preprint is not automatically a top-conference publication; formal venues are stated only where verified. Mutable main-branch links must be pinned to commits for experiments; the verification date is not a software version lock.

**[R01]** Kim, Kong, Son. Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech. ICML 2021.

<https://proceedings.mlr.press/v139/kim21f.html>

**[R02]** VITS authors. Official implementation: models.py and training code.

<https://github.com/jaywalnut310/vits>

**[R03]** Kong et al. VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial Learning and Architecture Design. Interspeech 2023.

<https://www.isca-archive.org/interspeech_2023/kong23_interspeech.html>

**[R04]** Li et al. StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models. NeurIPS 2023.

<https://arxiv.org/abs/2306.07691>

**[R05]** StyleTTS 2 authors. Official implementation.

<https://github.com/yl4579/StyleTTS2>

**[R06]** Du et al. CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models. Technical report, 2024.

<https://arxiv.org/abs/2412.10117>

**[R07]** Du et al. CosyVoice 3: Towards In-the-Wild Speech Generation via Scaling-up and Post-training. Technical report, 2025.

<https://arxiv.org/abs/2505.17589>

**[R08]** FunAudioLLM. Fun-CosyVoice3-0.5B-2512: official configuration.

<https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512/blob/main/cosyvoice3.yaml>

**[R09]** VoxCPM2 authors. VoxCPM2 Technical Report. 2026.

<https://arxiv.org/abs/2606.06928>

**[R10]** OpenBMB. VoxCPM2 AudioVAE implementation and checkpoint configuration.

<https://github.com/OpenBMB/VoxCPM/blob/main/src/voxcpm/modules/audiovae/audio_vae_v2.py> <https://huggingface.co/openbmb/VoxCPM2/raw/main/config.json>

**[R11]** Qwen team. Qwen3-TTS Technical Report. 2026.

<https://arxiv.org/abs/2601.15621>

**[R12]** Qwen team. Official Qwen3-TTS repository and Tokenizer-12Hz release documentation.

<https://github.com/QwenLM/Qwen3-TTS>

**[R13]** VoXtream2 authors. VoXtream2: Full-Stream TTS with Dynamic Speaking Rate Control. Preprint, 2026.

<https://arxiv.org/abs/2603.13518>

**[R14]** IndexTTS2 authors. IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech. Preprint, 2025.

<https://arxiv.org/abs/2506.21619>

**[R15]** Chen et al. WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing. 2021/2022.

<https://arxiv.org/abs/2110.13900>

**[R16]** Qwen team. Qwen3-1.7B model card and configuration.

<https://huggingface.co/Qwen/Qwen3-1.7B> <https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json>

**[R17]** Ho, Jain, Abbeel. Denoising Diffusion Probabilistic Models. NeurIPS 2020.

<https://arxiv.org/abs/2006.11239>

**[R18]** Lipman et al. Flow Matching for Generative Modeling. ICLR 2023.

<https://arxiv.org/abs/2210.02747>

**[R19]** Chen et al. Neural Ordinary Differential Equations. NeurIPS 2018.

<https://arxiv.org/abs/1806.07366>

**[R20]** Papamakarios et al. Normalizing Flows for Probabilistic Modeling and Inference. JMLR 2021.

<https://arxiv.org/abs/1912.02762>

**[R21]** Stavrou et al. Information Nonanticipative Rate Distortion Function and Its Applications. 2014.

<https://arxiv.org/abs/1405.1593>

**[R22]** Angelopoulos, Bates. A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. 2021.

<https://arxiv.org/abs/2107.07511>

**[R23]** Angelopoulos et al. Conformal Risk Control. Preprint first released 2022.

<https://arxiv.org/abs/2208.02814>

**[R24]** Xu, Karampatziakis, Mineiro. Active, Anytime-Valid Risk Controlling Prediction Sets. NeurIPS 2024.

<https://arxiv.org/abs/2406.10490>

**[R25]** Peyre, Cuturi. Computational Optimal Transport. 2019.

<https://arxiv.org/abs/1803.00567>

**[R26]** Tsitsiklis, Van Roy. Optimal Stopping of Markov Processes: Hilbert Space Theory, Approximation Algorithms, and an Application to Pricing High-Dimensional Financial Derivatives. IEEE TAC, 1999.

<https://web.stanford.edu/~bvr/pubs/stop.pdf>

**[R27]** Ito, Johnson. The LJ Speech Dataset. Official dataset page.

<https://keithito.com/LJ-Speech-Dataset/>

**[R28]** Koizumi et al. LibriTTS-R. Official OpenSLR release.

<https://openslr.org/141/>

**[R29]** Shi et al. AISHELL-3. Official OpenSLR release.

<https://openslr.org/93/>

**[R30]** Amphion/Emilia authors. Emilia data preparation and dataset licensing documentation.

<https://github.com/open-mmlab/Amphion/tree/main/preprocessors/Emilia>

**[R31]** ByteDance Speech. Seed-TTS-Eval official benchmark and evaluation code.

<https://github.com/BytedanceSpeech/seed-tts-eval>

**[R32]** Wen et al. X2Streaming-TTS: Causal Token-Level Text-to-Speech from Streaming Text with Speech-State Inheritance. Preprint, 2026.

<https://arxiv.org/abs/2608.18661>

**[R33]** Feng, Yoshimoto. Llama-VITS: Enhancing TTS Synthesis with Semantic Awareness. LREC-COLING 2024.

<https://aclanthology.org/2024.lrec-main.931/>

**[R34]** Dekel et al. Speak While You Think: Streaming Speech Synthesis During Text Generation. ICASSP 2024.

<https://research.ibm.com/publications/speak-while-you-think-streaming-speech-synthesis-during-text-generation>

**[R35]** Chen et al. F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching. ACL 2025.

<https://aclanthology.org/2025.acl-long.313/> <https://github.com/SWivid/F5-TTS>

**[R36]** Mehta et al. Matcha-TTS: A Fast TTS Architecture with Conditional Flow Matching. ICASSP 2024.

<https://arxiv.org/abs/2309.03199>

**[R37]** Niu et al. Semantic-VAE: Semantic-Alignment Latent Representation for Better Speech Synthesis. Preprint 2025; revised 2026.

<https://arxiv.org/abs/2509.22167>

**[R38]** NeurIPS. 2026 Reviewer Guidelines. Official guidance, accessed 2026-10-27.

<https://neurips.cc/Conferences/2026/ReviewerGuidelines>

**[R39]** McAuliffe et al. Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi. Interspeech 2017.

<https://www.isca-archive.org/interspeech_2017/mcauliffe17_interspeech.html>

**[R40]** Stephenson et al. Alternate Endings: Improving Prosody for Incremental Neural TTS with Predicted Future Text Input. Interspeech 2021.

<https://www.isca-archive.org/interspeech_2021/stephenson21_interspeech.html>

**[R41]** Nielsen, Sun. On the Chain Rule Optimal Transport Distance. Preprint 2018; revised 2020.

<https://arxiv.org/abs/1812.08113>

**[R42]** Liu et al. Prosodic Boundary-Aware Streaming Generation for LLM-Based TTS with Streaming Text Input. Preprint, 2026.

<https://arxiv.org/abs/2603.06444>

**[R43]** Verdini et al. Controlling Speaking Rate in Autoregressive TTS via Activation Steering. Preprint, September 2026; author-reported SLT 2026 acceptance.

<https://arxiv.org/abs/2609.33810>

