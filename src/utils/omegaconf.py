from omegaconf import OmegaConf


def register_resolvers():
    OmegaConf.register_new_resolver("resolve_path", lambda x: Path(x).resolve())
    OmegaConf.register_new_resolver("resolve_list", lambda x: [Path(item).resolve() for item in x])
    OmegaConf.register_new_resolver("resolve_dict", lambda x: {key: Path(value).resolve() for key, value in x.items()})
    OmegaConf.register_new_resolver("resolve_tuple", lambda x: tuple(Path(item).resolve() for item in x))
    OmegaConf.register_new_resolver("not", lambda boolean: not boolean)

    OmegaConf.register_new_resolver("equal", lambda arg1, arg2: arg1 == arg2)

    def conditional_resolver(condition, arg1, arg2):
        return arg1 if condition else arg2

    OmegaConf.register_new_resolver("ifelse", conditional_resolver)
