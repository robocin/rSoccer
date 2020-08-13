import gym
from gym_ssl.grsim_ssl.grSimClient import grSimClient

class GrSimSSLEnv(gym.Env):
    def __init__(self):
        self.client = grSimClient()
        self.action_space = None
        self.observation_space = None

    def step(self, action):
        # Envia ações do step para o grSim
        commands = self._getCommands(action) #commands são as ações do agente mais as ações do ia
        commandsPacket = self.__genCommandsPacket(commands) #gera pacote protobuf a partir de commands
        self.client.send(commandsPacket)

        # Recebe dados de visão pós ações do step
        visionData = self.client.receive()
        state = self._parseVision(visionData)

        # Calcula a recompensa e verifica fim de episodio
        reward, done = self._calculateRewards(visionData)

        return state, reward, done, {}

    def reset(self):
        # Place robots on initial positions
        initialPositions = self._getPositions() #get initial positions
        replacementPacket = self.__genReplacementPacket(initialPositions) #generate replacement packet 
        self.client.send(replacementPacket)

        # Recebe dados de visão pós reposicionamento
        visionData = self.client.receive()
        state = self._parseVision(visionData)

        return state

    # TODO a partir de uma lista de commands gerar o pacote de comando no formato da mensagem do pb
    def __genCommandsPacket(self, commands):
        raise NotImplementedError

    # TODO a partir de uma lista de posições gerar o pacote de replacement no formato da mensagem do pb
    def __genReplacementPacket(self, positions):
        raise NotImplementedError
    
    # Para ser definido pelos envs filho, deve transformar a ação (como definida pelo ambiente) no formato recebido pelo __genCommandsPacket(), tambem deve aqui ser definidas as ações do oponente
    def _getCommands(self, action):
        raise NotImplementedError

    # Para ser definido pelos envs filho, deve retornar as posições iniciais no formato recebido pelo __genReplacementPackage()
    def _getPositions(self):
        raise NotImplementedError

    # Para ser definido pelos envs filho, deve a partir do pacote de visão retornar o estado no formado como definido pelo ambiente
    def _parseVision(self, visionData):
        raise NotImplementedError

    # Para ser definidopelos envs filho, deve a partir da mensagem de visão calcular a recompensa e definir se chegou ao fim do episodio
    def _calculateRewards(self, visionData):
        raise NotImplementedError

