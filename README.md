# Semantic Navigation

Project Name: Semantic Navigation

## Dependencies:
included other github repos, list all of them with URL and their license
- ConceptGraph
    - https://github.com/concept-graphs/concept-graphs
    - MIT License
- SkeletonFinder
    - https://github.com/xchencq/3D_Sparse_Skeleton
    - GNU GENERAL PUBLIC LICENSE

## Execution Scope
This project explores a multidisciplinary approach to robotic navigation, integrating spatial computing, control theory, and semantic understanding. Our goal is to enable robots to interpret and execute commands given in natural language. We developed a semantic navigation framework that is independent of spatial representation and robot configuration. By combining semantic features from foundation models with sparse skeleton graph construction, we provide a robust system for ground and aerial robots to navigate complex environments based on human queries.

The project focuses on creating a general architecture for commanding robots with natural language by building topological representations of environments, assigning semantic labels, and leveraging reinforcement learning for local navigation policies. The system has been implemented in a photorealistic simulation environment, with an emphasis on seamless transfer to real-world deployment, ensuring adaptability and scalability.

## Solution Description
Key Features:
- Post-Processing Algorithm for Graphs: A novel enhancement method for sparse skeleton graphs, producing cleaner and more structured representations.
- Raycasting-Based Path Generation: Ensures safe path generation for ground robots, considering obstacles and environmental constraints.
- Efficient Exploration Algorithm: Facilitates exploration of unknown environments while incrementally constructing the topological graph.
- Semantic Partitioning and Labeling: Graphs are divided into semantically meaningful regions (e.g., rooms) and assigned contextual labels using a Large Language Model.
- Reinforcement Learning for Local Policies: Ensures navigation between waypoints, making the system invariant to robot configurations.
-  Photorealistic Simulation and Real-World Transfer: Rapid prototyping in simulation environments with seamless deployment to physical systems.

Own Contributions:
- Development of the skeleton graph enhancement algorithm.
- Creation of the raycasting-based safe path generation method.
- Novel exploration algorithm for unknown environments.
- Integration of semantic understanding and language-based commands into navigation.
- Demonstrating real-world scalability through simulated and real-world experiments.

## User Value
The project empowers users, including researchers and developers, to implement flexible and robust navigation systems capable of interpreting natural language queries. By integrating semantic understanding with navigation, the system enables robots to perform tasks such as identifying and navigating to specific objects or regions in the environment. The modular and extensible architecture reduces development cycles and allows for future improvements, making it a valuable resource for advancing robotic autonomy in various real-world applications.

## Business Value
The framework offers significant value for industries requiring autonomous navigation solutions, such as logistics, healthcare, and service robotics. By enabling natural language-based commands, the system reduces the complexity of human-robot interaction, making it more accessible to non-technical users. The focus on scalability, modularity, and adaptability ensures that the framework can be integrated into diverse robotic platforms, accelerating development cycles and reducing deployment costs.

## Requested License
We propose the use of the MIT License for this project. The MIT License is a permissive license that encourages open collaboration and broad adoption while ensuring compliance with any dependencies used in the project.

## Attachments:
- [Video Demonstration](https://youtu.be/D2kV2Eo8oFg)
