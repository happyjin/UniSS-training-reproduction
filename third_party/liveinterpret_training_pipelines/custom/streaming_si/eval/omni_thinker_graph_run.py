"""Plain runner (same CLI as omni_shared_run.py) that installs the thinker decode-step graph after model load.
No timing wrappers, so end-to-end numbers are not perturbed by extra synchronisation."""
import omni_shared_run as shared
import omni_thinker_graph


def main():
    import omni_simuleval_wrap  # noqa: F401  (same hook order as the other runners)
    import src.inference.infer_omni_talker_tts as tts
    cls = tts.OmniTalkerSemanticTTSInference
    original_load = cls.from_pretrained

    def load(_cls, *args, **kwargs):
        model = original_load(*args, **kwargs)
        omni_thinker_graph.maybe_install(model)
        return model

    cls.from_pretrained = classmethod(load)
    shared.main()


if __name__ == '__main__':
    main()
