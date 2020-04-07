2019-12-22
移动系统环境配置至environconfig目录
将deepsight变为应用
启动只需启动ai应用即可: python manage.py ai



initialized 2019-12-19 初始化项目

Django参数及配置说明
1. Django的环境配置以灵活为主要目的，所有的配置都可以放入代码库中，settings目录下，例如基础配置文件base.py，本地配置文件local.py，生产环境配置文件production.py等
   
2. 启动命令必须按照如下参数启动
python manage.py runserver --settings=deepsight.settings.xxx
xxx即为settings目录下的除base之外的配置文件名(local,staging,testing,production等)

3. 非生产环境(production)的SECRET_KEY可以配置在配置文件中
   
4. 生产环境(production)的SECRET_KEY必须配置在单独的json文件中，此json文件不可以放入git代码库中，使用方法为
   A. 创建某json配置文件，例如/etc/django/config.json，内容包含敏感SECRET_KEY内容
   {
       "SECRET_KEY" : "2d$rz9bgy0=&1k4(nq6cu@s0wdc4qn3df_j48n*p6)k@zzk0-)"
   }
   B. 设置环境参数DEEPSIGHT_CONFIG， 例如export DEEPSIGHT_CONFIG="/etc/django/config.json"，注意此文件的读写属性
   C. 启动python manage.py runserver --settings=deepsight.settings.production
