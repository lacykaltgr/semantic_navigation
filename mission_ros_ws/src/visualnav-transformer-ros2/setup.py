from setuptools import setup, find_packages

package_name = 'visnav_transformer'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']), # install every folder except test
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lacykaltgr',
    maintainer_email='freundl0509@gmail.com',
    description='Visual Navigation Transformer',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'explore = visualnav_transformer.deployment.explore:main',
            'create_topomap = visualnav_transformer.deployment.create_topomap:main',
        ],
    },
)
