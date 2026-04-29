from PIL import Image
import numpy as np


class MazeObject:
    """
    Base class for any object or entity found within the maze.
    """

    def __init__(self, type, pos):
        self.type = type
        self.pos = pos
        self.image = None
        self.max_steps = 200
        self.current_step = 0

    def can_overlap(self):
        """
        Boolean indicating whether robot and object can occupy the same cell.
        """
        return False

    def get_camera_view(self):
        """
        Returns the 'camera view' of the robot when facing cell.
        """
        return self.image

    def can_be_killed_by_sword(self):
        """
        Boolean indicating whether object can be killed (i.e. removed from the environment)
        by robot using a sword.
        """
        return False

    def can_be_killed_by_bow(self):
        """
        Boolean indicating whether object can be killed (i.e. removed from the environment)
        by robot using a bow.
        """
        return False


class Target(MazeObject):
    """
    Robot's target or goal location.
    """

    def __init__(self, pos):
        super().__init__("target", pos)
        # Target appears as greyscale green image - update to door image?
        self.image = 146 * np.ones((20, 20))

    def can_overlap(self):
        return True


class Wall(MazeObject):
    """
    The maze walls, robot cannot walk through them.
    """

    def __init__(self, pos):
        super().__init__("wall", pos)
        # Walls appear as black image.
        self.image = np.zeros((20, 20))


class Orc(MazeObject):
    """
    Orc creatures.
    Can be killed with a sword but not a bow as too strong.
    Will kill robot if overlapped.
    """

    def __init__(self, pos, image_id):
        super().__init__("orc", pos)
        assert image_id >= 0 and image_id <= 99
        im = Image.open("images/orc/orc_{}.png".format(str(image_id).zfill(3)))
        self.image = np.array(im)
        im.close()

    def can_overlap(self):
        return True

    def can_be_killed_by_sword(self):
        return True


class Wingedbat(MazeObject):
    """
    Winged bat creatures.
    Can be killed with a bow but not a sword as flying.
    Will kill robot if overlapped.
    """

    def __init__(self, pos, image_id):
        super().__init__("wingedbat", pos)
        assert image_id >= 0 and image_id <= 99
        im = Image.open(
            "images/wingedbat/wingedbat_{}.png".format(str(image_id).zfill(3))
        )
        self.image = np.array(im)
        im.close()

    def can_overlap(self):
        return True

    def can_be_killed_by_bow(self):
        return True


class Lizard(MazeObject):
    """
    Lizard creatures.
    Can be killed with both bow and sword.
    Will kill robot if overlapped.
    """

    def __init__(self, pos, image_id):
        super().__init__("lizard", pos)
        assert image_id >= 0 and image_id <= 99
        im = Image.open("images/lizard/lizard_{}.png".format(str(image_id).zfill(3)))
        self.image = np.array(im)
        im.close()

    def can_overlap(self):
        return True

    def can_be_killed_by_bow(self):
        return True

    def can_be_killed_by_sword(self):
        return True


# ─── Behaviour-cluster entities (Task 2 → Task 3 integration) ─────────────────

class TankEntity(MazeObject):
    """
    Tank-type entity — high strength, blocks the robot's path.
    Weakness: flee  (direct confrontation is futile; running away disengages it)
    Resists:  bow, sword
    Camera image: dark grey (greyscale 60)
    """
    def __init__(self, pos):
        super().__init__("tank", pos)
        self.image = np.full((20, 20), 60, dtype=np.float64)


class FlyingEntity(MazeObject):
    """
    Flying-type entity — airborne, can chase a fleeing robot.
    Weakness: bow    (ranged attack reaches airborne targets)
    Resists:  flee (gives chase), sword (melee can't reach)
    Camera image: medium grey (greyscale 150)
    """
    def __init__(self, pos):
        super().__init__("flying", pos)
        self.image = np.full((20, 20), 150, dtype=np.float64)


class SmartEntity(MazeObject):
    """
    Smart-type entity — high intelligence, predicts robot behaviour.
    Weakness: sword  (up-close melee before it can react)
    Resists:  flee (intercepts), bow (dodges ranged shots)
    Camera image: light grey (greyscale 210)
    """
    def __init__(self, pos):
        super().__init__("smart", pos)
        self.image = np.full((20, 20), 210, dtype=np.float64)
