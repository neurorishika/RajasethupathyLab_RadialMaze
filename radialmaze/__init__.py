import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'environments',
        'rdp_client',
    },
    submod_attrs={
        'environments': [
            'RadialMaze',
        ],
        'rdp_client': [
            'unlock_and_unzip_file',
            'zip_and_lock_folder',
        ],
    },
)

__all__ = ['RadialMaze', 'environments', 'rdp_client', 'unlock_and_unzip_file',
           'zip_and_lock_folder']
