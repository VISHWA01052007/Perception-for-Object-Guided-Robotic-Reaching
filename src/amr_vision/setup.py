from setuptools import setup

package_name = 'amr_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vishwa',
    maintainer_email='vishwa@todo.todo',
    description='AMR Vision perception nodes',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'red_detector = amr_vision.red_detector:main',
            'eye_in_sky = amr_vision.eye_in_sky:main',
        ],
    },
)