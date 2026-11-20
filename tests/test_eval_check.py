
import inspect
import train_ippo
import tools.evaluate_marl as em
print('train_ippo args:', inspect.signature(train_ippo.train_ippo))
print('evaluate_marl args:', inspect.signature(em.evaluate_policy))
