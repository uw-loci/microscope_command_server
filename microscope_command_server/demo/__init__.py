"""Demo microscope: the command server with no Micro-Manager and no hardware.

``microscope-command-server --demo`` swaps the Pycro-Manager ``Core`` for
:class:`~microscope_command_server.demo.virtual_microscope.DemoCore`, an in-memory
stand-in whose devices carry the same names as Micro-Manager's own Demo adapter
(``DCam``, ``DStage``, ``XYStage``, ``DLamp``). Everything above the core -- the
camera and stage classes in ``microscope_control``, every socket handler, the
acquisition workflow, autofocus -- is the production code. Only the device layer
is simulated, which is the same split Micro-Manager itself makes for its demo
configuration.
"""
