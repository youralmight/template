# this is ok
python src/main/cli.py fit --model model.LitAutoEncoder --model.fake_argument fake --data PLMnist --ckpt_path exps
# this is not ok
python src/main/cli.py fit --model.fake_argument fake --model model.LitAutoEncoder --data PLMnist

# show available arguments
python src/main/cli.py fit --help

# trainer args are same as the parameters of the class Trainer
python src/main/cli.py fit --model model.LitAutoEncoder --model.fake_argument fake --data PLMnist --trainer.default_root_dir exps --trainer.log_dir exps
'''
--trainer plenty of arguments
--model CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE
--data CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE
--optimizer CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE
--lr_scheduler.help CLASS_PATH_OR_NAME
'''


'''
model:
  class_path: mycode.mymodels.MyModel
  init_args:
    decoder_layers: [2, 4]
    encoder_layers: 12

python main.py fit --model.class_path mycode.mymodels.MyModel --model.init_args.decoder_layers [2,4] --model.init_args.encoder_layers 12
'''