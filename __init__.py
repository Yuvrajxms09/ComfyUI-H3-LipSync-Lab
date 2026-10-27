"""ComfyUI entrypoint for the H3 lip-sync tracing nodes.

Pytest imports a hyphenated custom-node directory's root ``__init__.py`` as a
top-level module during collection. In that context there is no package anchor
for a relative import, and the core metric tests do not need ComfyUI node
registration. ComfyUI itself loads this file as a package, so registration is
performed only in that real runtime context.
"""

if __package__:
    from .nodes import comfy_entrypoint

    __all__ = ["comfy_entrypoint"]
else:
    __all__ = []
