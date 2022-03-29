# pip install coloredlogs

import logging
import coloredlogs

lg = logging.getLogger('your-module')

coloredlogs.DEFAULT_FIELD_STYLES = {'asctime': {'color': 'green'}, 'hostname': {'color': 'magenta'},
                                    'levelname': {'bold': True, 'color': (170, 170, 170)}, 'name': {'color': 'blue'},
                                    'programname': {'color': (170, 170, 170)}, 'username': {'color': 'yellow'}}
coloredlogs.install(logger=lg, level='DEBUG', fmt='%(programname)s:%(message)s', isatty=True)

lg.debug("this is a debugging message")
lg.info("this is an informational message")
lg.warn("this is a warning message")
lg.warning("this is a warning message")
lg.error("this is an error message")
lg.critical("this is a critical message")
