
# Global dependencies dictionary
dependencies = dict()


def set_dependency(**kwargs):
    global dependencies  # Declare that we're using the global variable
    dependencies.update(kwargs)
