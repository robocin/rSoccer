import gym
from gym_ssl.grsim_ssl.grSimClient import grSimClient

class GrSimSSLEnv(gym.Env):
    def __init__(self):
        self.client = grSimClient()
        self.state = None
        self.action_space = None
        self.observation_space = None

    def step(self, action):
        # Envia ações do step para o grSim
        commands = self._getCommands(action) #commands são as ações do agente mais as ações do ia
        self.__sendCommandsPacket(commands) #gera pacote protobuf a partir de commands e envia

        # Recebe dados de visão pós ações do step
        self.state = self.__getState()
        observation = self._parseState()

        # Calcula a recompensa e verifica fim de episodio
        reward, done = self._calculateRewards()

        return observation, reward, done, {}

    def reset(self):
        # Place robots on initial positions
        initialFormation = self._getFormation() #get initial positions
        self.__sendReplacementPacket(initialFormation) #generate and send replacement packet 

        # Recebe dados de visão pós reposicionamento
        self.state = self.__getState()
        observation = self._parseState()

        return observation

    # TODO a partir de uma lista de commands gerar o pacote de comando no formato da mensagem do pb e envia
    def __sendCommandsPacket(self, commands):
        raise NotImplementedError

    # TODO a partir de uma lista de posições gerar o pacote de replacement no formato da mensagem do pb e envia
    def __sendReplacementPacket(self, positions):
        raise NotImplementedError
    
    # TODO
    def __getState(self):
        raise NotImplementedError

    # Para ser definido pelos envs filho, deve transformar a ação (como definida pelo ambiente) no formato recebido pelo __genCommandsPacket(), tambem deve aqui ser definidas as ações do oponente
    def _getCommands(self, action):
        raise NotImplementedError

    # Para ser definido pelos envs filho, deve retornar as posições iniciais no formato recebido pelo __genReplacementPackage()
    def _getFormation(self):
        raise NotImplementedError

    # Para ser definido pelos envs filho, deve a partir do pacote de visão retornar o estado no formado como definido pelo ambiente
    def _parseState(self):
        raise NotImplementedError

    # Para ser definidopelos envs filho, deve a partir da mensagem de visão calcular a recompensa e definir se chegou ao fim do episodio
    def _calculateRewards(self):
        raise NotImplementedError

