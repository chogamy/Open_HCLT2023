from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf


def get_configurable_parameters(args):
    if args.config is not None:
        config = OmegaConf.load(args.config)
    else:
        config = OmegaConf.create({})
            
    # keep track of the original config file because it will be modified
    config_original: DictConfig = config.copy()

    if args.model is not None:
        config = OmegaConf.merge(config, OmegaConf.load(args.model))
        
    if args.data is not None:
        config = OmegaConf.merge(config, OmegaConf.load(args.data))
        
    if args.trainer is not None:
        config = OmegaConf.merge(config, OmegaConf.load(args.trainer))
            
    if args.seed is None and config.seed is None:
        config.seed = 1235
    elif args.seed is not None:
        config.seed = int(args.seed)

    if args.mode is not None:
        config.mode = args.mode
    
    if args.model_name_or_path is not None:
        ## PLM name issue ##################################################
        if args.model_name_or_path == "electra-base-discriminator":
            args.model_name_or_path = "google/electra-base-discriminator"
        if args.model_name_or_path == "koelectra-small-v3-discriminator":
            args.model_name_or_path = "monologg/koelectra-small-v3-discriminator"
        if args.model_name_or_path == "koelectra-base-v3-discriminator":
            args.model_name_or_path = "monologg/koelectra-base-v3-discriminator"
        if args.model_name_or_path == "kobert-base-v1":
            args.model_name_or_path = "skt/kobert-base-v1"
        ## PLM name issue ##################################################
        config.model_name_or_path = args.model_name_or_path
        config.model.init_args.model_name_or_path = args.model_name_or_path
        config.data.model_name_or_path = args.model_name_or_path
        
    # PreTrain
    if config.model.class_path == "adb.adb.ADB":
        plm = config.model.init_args.model_name_or_path.split("_")[1]
        ## PLM name issue ##################################################
        if plm == 'electra-base-discriminator':
            plm = 'google/electra-base-discriminator'
        if plm == 'koelectra-small-v3-discriminator':
            plm = 'monologg/koelectra-small-v3-discriminator'
        if plm == 'koelectra-base-v3-discriminator':
            plm = 'monologg/koelectra-base-v3-discriminator'
        if plm == 'kobert-base-v1':
            plm = 'skt/kobert-base-v1'
        ## PLM name issue ##################################################
        config.data.model_name_or_path = plm    
        
    # K_1 setting
    if config.model.class_path == "K_1_way.K_1_way.K_1_way":
        config.data.k_1 = True
        
    if config.data.known_cls_ratio is None:
        config.data.known_cls_ratio = float(args.known_cls_ratio)
        
    model_name = config.model.class_path.split(".")[-1]
    plm = config.data.model_name_or_path
    ## PLM name issue ##################################################
    if plm == 'google/electra-base-discriminator':
        plm = 'electra-base-discriminator'
    if plm == 'monologg/koelectra-small-v3-discriminator':
        plm = 'koelectra-small-v3-discriminator'
    if plm == 'monologg/koelectra-base-v3-discriminator':
        plm = 'koelectra-base-v3-discriminator'
    if plm == 'skt/kobert-base-v1':
        plm = 'kobert-base-v1'
    ## PLM name issue ##################################################
    file_name = f"{model_name}_{plm}_{config.data.dataset}{config.data.known_cls_ratio}_{config.seed}"
    config.trainer.callbacks[1].init_args.filename = file_name
    
    # sampler
    if args.sampler is not None:
        sampler = OmegaConf.load(args.sampler)
        config.model.init_args.sampler = sampler.sampler
        config.model.init_args.sampler_config = sampler.sampler_config
    

    # Project Configs
    project_path = Path(
        "./trainer_logs"
    )  # Path(config.project.path) / config.model.name / config.dataset.name
    project_path.mkdir(parents=True, exist_ok=True)
    
    # (project_path / "weights").mkdir(parents=True, exist_ok=True)

    # write the original config for eventual debug (modified config at the end of the function)
    # (project_path / "config_original.yaml").write_text(OmegaConf.to_yaml(config_original))
    # config.project.path = str(project_path)

    config.trainer.default_root_dir = str(project_path)
    
    # if weight_file:
    #     config.trainer.resume_from_checkpoint = weight_file
    # config_name = f"{}.yaml"
    
    (project_path / f"{file_name}.yaml").write_text(OmegaConf.to_yaml(config))
    
    return config
