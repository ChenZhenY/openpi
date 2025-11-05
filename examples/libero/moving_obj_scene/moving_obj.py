import numpy as np

class MovingObjectEnvWrapper:
    """Wrapper to make objects move in the environment."""
    
    def __init__(self, env, object_name="black_bowl", 
                 movement_axis='x', movement_range=0.15, movement_speed=0.01):
        """
        Args:
            env: The base LIBERO environment
            object_name: Name of the object to move
            movement_axis: 'x' or 'y' for movement direction
            movement_range: Total distance to move (meters)
            movement_speed: Speed of movement per step (meters/step)
        """
        self.env = env
        self.object_name = object_name
        self.movement_axis = 0 if movement_axis == 'x' else 1  # 0 for x, 1 for y
        self.movement_range = movement_range
        self.movement_speed = movement_speed
        
        # Movement state
        self.initial_pos = None
        self.direction = 1  # 1 for forward, -1 for backward
        self.current_offset = 0.0
        
    def reset(self):
        """Reset the environment and record initial object position."""
        obs = self.env.reset()
        
        # Get the initial position of the object
        if hasattr(self.env, 'env'):
            env_obj = self.env.env
        else:
            env_obj = self.env
            
        try:
            if self.object_name in env_obj.obj_body_id:
                body_id = env_obj.obj_body_id[self.object_name]
                self.initial_pos = env_obj.sim.data.body_xpos[body_id].copy()
                self.current_offset = 0.0
                self.direction = 1
        except AttributeError:
            # If we can't access the object, just continue
            pass
            
        return obs
    
    def step(self, action):
        """Step the environment and update object position."""
        # Step the base environment
        obs, reward, done, info = self.env.step(action)
        
        # Update object position
        self._update_object_position()
        
        return obs, reward, done, info
    
    def _update_object_position(self):
        """Update the moving object's position."""
        if self.initial_pos is None:
            return
            
        # Get environment object
        if hasattr(self.env, 'env'):
            env_obj = self.env.env
        else:
            env_obj = self.env
            
        try:
            if self.object_name not in env_obj.obj_body_id:
                return
                
            # Update offset
            self.current_offset += self.direction * self.movement_speed
            
            # Reverse direction if we hit the range limits
            if abs(self.current_offset) >= self.movement_range / 2:
                self.direction *= -1
                self.current_offset = np.clip(
                    self.current_offset, 
                    -self.movement_range / 2, 
                    self.movement_range / 2
                )
            
            # Calculate new position
            new_pos = self.initial_pos.copy()
            new_pos[self.movement_axis] += self.current_offset
            
            # Get the object and update its position via joint
            obj = env_obj.objects_dict[self.object_name]
            if obj.joints:
                # Get current joint state (position + quaternion)
                joint_name = obj.joints[-1]  # Free joint
                qpos_addr = env_obj.sim.model.get_joint_qpos_addr(joint_name)
                current_qpos = env_obj.sim.data.qpos[qpos_addr[0]:qpos_addr[1]].copy()
                
                # Update position, keep orientation
                new_qpos = current_qpos.copy()
                new_qpos[:3] = new_pos  # Update x, y, z position
                
                # Set the new joint position
                env_obj.sim.data.set_joint_qpos(joint_name, new_qpos)
                
                # Forward the simulation to apply changes
                env_obj.sim.forward()
                
        except (AttributeError, KeyError) as e:
            # Silently handle if object doesn't exist or isn't movable
            pass
    
    def __getattr__(self, name):
        """Forward all other attributes to the wrapped environment."""
        return getattr(self.env, name)