#pragma once
//
// Header file for UDP communication
//
#include "settings.h"
#include <sys/types.h> 
#include <sys/socket.h> 
#include <arpa/inet.h> 
#include <netinet/in.h> 

// PORT hardcoded for the time beign

#define UDPCOMM_PORT 12099

struct UDPCOMM {
  int sockfd; 
  struct sockaddr_in servaddr, cliaddr; 
};

void UDPCommInit (UDPCOMM *udp, SETTINGS *set);

void UDPPassKeyPress (UDPCOMM *udp, char key);
void UDPGetKeyPress (UDPCOMM *udp, char *key);


